"""Drain3-based log template parser with persistence support."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import re

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
from tqdm import tqdm

from constants import LOG_ROOT, PROJECT_ROOT, SESSION


@dataclass
class LogFormat:
    """Log format specification."""
    pattern: str
    headers: List[str]
    regex: re.Pattern

    @classmethod
    def from_pattern(cls, pattern: str) -> 'LogFormat':
        """Create LogFormat from pattern string."""
        headers = []
        splitters = re.split(r'(<[^<>]+>)', pattern)
        
        # Build regex pattern
        regex_parts = []
        for i, splitter in enumerate(splitters):
            if i % 2 == 0:
                regex_parts.append(re.sub(r'\s+', r'\s+', splitter))
            else:
                header = splitter.strip('<>')
                regex_parts.append(f'(?P<{header}>.*?)')
                headers.append(header)
                
        regex = re.compile('^' + ''.join(regex_parts) + '$')
        return cls(pattern, headers, regex)


class Drain3Parser:
    """Log template parser using Drain3 algorithm.
    
    Supports:
    - Template mining with configurable parameters
    - State persistence
    - Incremental parsing
    - Column removal and cleaning
    """
    
    DEFAULT_LOG_FORMAT = (
        "<Logrecord> <Date> <Time> <Pid> <Level> <Component> [<ADDR>] <Content>"
    )
    
    def __init__(
        self,
        config_path: Path,
        persistence_dir: Path,
        log_format: Optional[str] = None
    ):
        """Initialize parser with configuration and persistence directory.
        
        Args:
            config_path: Path to Drain3 config file
            persistence_dir: Directory for storing parser state
            log_format: Optional custom log format pattern
        """
        self.logger = self._setup_logger()
        self.config = self._load_config(config_path)
        self.persistence_dir = self._setup_persistence_dir(persistence_dir)
        self.log_format = LogFormat.from_pattern(log_format or self.DEFAULT_LOG_FORMAT)
        
        # Initialize parser
        persistence_file = self.persistence_dir / "persistence"
        self.parser = TemplateMiner(
            persistence_handler=FilePersistence(str(persistence_file)),
            config=self.config
        )
        
        # Load existing state if available
        self.needs_training = not self._load_state(persistence_file)

    def _setup_logger(self) -> logging.Logger:
        """Configure logging for parser."""
        logger = logging.getLogger("drain")
        logger.setLevel(logging.DEBUG)
        
        # Create formatters and handlers
        fmt = logging.Formatter(
            f"%(asctime)s - %(name)s - {SESSION} - %(levelname)s: %(message)s"
        )
        
        # Console handler
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        console.setLevel(logging.DEBUG)
        
        # File handler
        file_handler = logging.FileHandler(Path(LOG_ROOT) / "drain.log")
        file_handler.setFormatter(fmt)
        file_handler.setLevel(logging.INFO)
        
        # Add handlers
        logger.addHandler(console)
        logger.addHandler(file_handler)
        
        logger.info(f"Logger initialized. Logs will be written to {LOG_ROOT}")
        return logger

    def _load_config(self, config_path: Path) -> TemplateMinerConfig:
        """Load Drain3 configuration from file."""
        config = TemplateMinerConfig()
        
        if not config_path.exists():
            self.logger.info("No config file found, using defaults")
        else:
            self.logger.info(f"Loading config from {config_path}")
            config.load(str(config_path))
            
        config.profiling_enabled = False
        return config

    def _setup_persistence_dir(self, base_dir: Path) -> Path:
        """Create and return persistence directory path."""
        dir_name = f"ibm_drain_depth-{self.config.drain_depth}_st-{self.config.drain_sim_th}"
        persist_dir = base_dir / dir_name
        
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Using persistence directory: {persist_dir}")
        
        return persist_dir

    def _load_state(self, persistence_file: Path) -> bool:
        """Load parser state from persistence file.
        
        Returns:
            True if state was loaded successfully
        """
        if not persistence_file.exists():
            self.logger.info(f"No persistence file found at {persistence_file}")
            return False
            
        self.logger.info("Loading existing parser state")
        self.parser.load_state()
        return True

    def parse_file(
        self,
        input_path: Path,
        remove_cols: Optional[List[int]] = None,
        clean: bool = False
    ) -> List[Any]:
        """Parse log file and extract templates.
        
        Args:
            input_path: Path to log file
            remove_cols: Column indices to remove
            clean: Whether to clean special characters
            
        Returns:
            List of extracted templates
        """
        self.logger.info(f"Parsing file: {input_path}")
        
            if remove_cols:
            self.logger.info(f"Removing columns: {remove_cols}")
            
        # Process file
        with open(input_path) as f:
            for line in tqdm(f, desc="Parsing logs"):
                line = line.strip()
                if remove_cols:
                    line = self._remove_columns(line, remove_cols, clean)
                self.parser.add_log_message(line)
                
        # Save state and templates
        self.parser.save_state("Parsing complete")
        self._save_templates()
        
        return self.parser.drain.clusters

    def parse_line(
        self,
        line: str,
        remove_cols: Optional[List[int]] = None,
        save_state: bool = False
    ) -> List[Any]:
        """Parse single log line.
        
        Args:
            line: Log line to parse
            remove_cols: Column indices to remove
            save_state: Whether to save parser state after parsing
            
        Returns:
            Updated template clusters
        """
        line = line.strip()
        if remove_cols:
            line = self._remove_columns(line, remove_cols)
            
        self.parser.add_log_message(line)
        
        if save_state:
            self.parser.save_state("Single line parsed")
            
        return self.parser.drain.clusters

    def _remove_columns(
        self,
        line: str,
        remove_cols: List[int],
        clean: bool = False
    ) -> str:
        """Remove specified columns from log line.
        
        Args:
            line: Input log line
            remove_cols: Column indices to remove
            clean: Whether to clean special characters
            
        Returns:
            Processed log line
        """
        tokens = line.split()
        kept_tokens = [
            token for i, token in enumerate(tokens)
            if i not in remove_cols
        ]
        
        result = ' '.join(kept_tokens)
        if clean:
            result = re.sub(r'[\*\.\?\+\$\^\[\]\(\)\{\}\|\\\/]', '', result)
            
        return result

    def _save_templates(self) -> None:
        """Save extracted templates to file."""
        template_file = self.persistence_dir / "templates.txt"
        
        with open(template_file, 'w') as f:
            for cluster in self.parser.drain.clusters:
                f.write(f"{cluster}\n")
                
        self.logger.info(f"Templates saved to {template_file}")

    def match(self, line: str) -> Dict[str, Any]:
        """Match log line against existing templates."""
        return self.parser.match(line)


if __name__ == "__main__":

    parser = Drain3Parser(
        config_path=Path(PROJECT_ROOT) / "conf/drain3.ini",
        persistence_dir=Path(PROJECT_ROOT) / "datasets/HDFS/persistences",
    )
    parser.logger.info("Testing program start.")
    remove_cols = [0, 1, 2, 3, 4]
    input_file = Path(PROJECT_ROOT) / "datasets/HDFS/HDFS.log"

    if parser.needs_training:
        # learn log events from raw log.
        parser.logger.info("Start training a new parser.")
        if not input_file.exists():
            parser.logger.error(
                "File %s not found. Please check the dataset folder" % input_file
            )
            sys.exit(1)
        parser.parse_file(input_path=input_file, remove_cols=remove_cols)

    pass
