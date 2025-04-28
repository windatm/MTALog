"""Base data loader for log analysis with parsing capabilities."""

import abc
import os
import sys
import time
from dataclasses import dataclass, field
from multiprocessing import Manager, Pool
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Callable, Tuple, Union
import logging

import numpy as np
from tqdm import tqdm

from data.parsers.Drain_IBM import Drain3Parser


@dataclass
class ParsingResult:
    """Container for parsing operation results."""
    log2temp: Dict[int, int] = field(default_factory=dict)
    templates: Dict[int, str] = field(default_factory=dict)
    time_taken: float = 0.0


def _async_parsing(parser: Drain3Parser, lines: List[Tuple[int, str]], log2temp: Dict[int, int]) -> None:
    """Parse log lines in parallel worker process.
    
    Args:
        parser: Initialized Drain3Parser
        lines: List of (log_id, log_line) tuples to parse
        log2temp: Shared dict for storing results
    """
    for log_id, line in lines:
        cluster = parser.match(line)
        log2temp[log_id] = cluster.cluster_id


class BasicDataLoader(abc.ABC):
    """Abstract base class for log data loaders with parsing functionality.
    
    Provides common methods for loading, parsing, and managing log data.
    Subclasses must implement abstract methods for specific log formats.
    """
    
    def __init__(self) -> None:
        """Initialize data loader with empty state."""
        # File paths
        self.in_file: Optional[Path] = None
        
        # Logging
        self.logger: Optional[logging.Logger] = None
        
        # Data structures
        self.block2emb: Dict[str, Any] = {}
        self.blocks: List[str] = []
        self.templates: Dict[int, str] = {}
        self.log2temp: Dict[int, int] = {}
        self.rex: List[str] = []
        self.remove_cols: List[int] = []
        
        # Mappings
        self.id2label: Dict[int, str] = {0: "Normal", 1: "Anomalous"}
        self.label2id: Dict[str, int] = {"Normal": 0, "Anomalous": 1}
        
        # Block data
        self.block_set: Set[str] = set()
        self.block2seqs: Dict[str, List[int]] = {}
        self.block2label: Dict[str, str] = {}
        self.block2eventseq: Dict[str, List[int]] = {}
        
        # Embeddings
        self.id2embed: Dict[int, np.ndarray] = {}
        self.semantic_repr_func: Optional[Callable[[Dict[int, str]], Dict[int, np.ndarray]]] = None

    @abc.abstractmethod
    def _load_raw_log_seqs(self) -> None:
        """Load raw log sequences from source file."""
        pass

    @abc.abstractmethod
    def _pre_process(self, line: str) -> str:
        """Pre-process a log line before parsing.
        
        Args:
            line: Raw log line
            
        Returns:
            Processed log line ready for parsing
        """
        pass

    @property
    def has_logger(self) -> bool:
        """Check if logger is configured."""
        return self.logger is not None
    
    def _ensure_logger(self) -> None:
        """Ensure logger is initialized before use."""
        if not self.has_logger:
            raise ValueError("Logger not initialized")
    
    def parse_by_IBM(
        self, 
        config_file: Union[str, Path], 
        persistence_folder: Union[str, Path], 
        core_jobs: int = 5
    ) -> None:
        """Parse logs using the IBM Drain algorithm.
        
        Args:
            config_file: Path to Drain3 configuration file
            persistence_folder: Directory for storing parser state
            core_jobs: Number of parallel jobs for parsing
        """
        self._ensure_logger()
        self._restore()
        
        # Convert to Path objects
        config_path = Path(config_file)
        persist_path = Path(persistence_folder)
        
        # Verify config file exists
        if not config_path.exists():
            self.logger.error(f"IBM Drain config file {config_path} not found")
            sys.exit(1)
            
        # Initialize parser
        parser = Drain3Parser(config_file=config_path, persistence_folder=persist_path)
        persist_path = parser.persistence_folder  # Get normalized path
        
        # Define persistence files
        log_event_seq_file = persist_path / "log_sequences.txt"
        log_template_mapping_file = persist_path / "log_event_mapping.dict"
        templates_embedding_file = persist_path / "templates.vec"
        
        start_time = time.time()
        
        # Train parser if needed
        if parser.to_update:
            self._train_parser(parser)
        
        # Load templates from trained parser
        self._load_templates_from_parser(parser)
        
        # Load or generate parsing results
        if self._check_parsing_persistences(log_template_mapping_file, log_event_seq_file):
            self.load_parsing_results(log_template_mapping_file, log_event_seq_file)
        else:
            self._generate_parsing_results(parser, log_template_mapping_file, log_event_seq_file, core_jobs)
        
        # Prepare semantic embeddings
        self._prepare_semantic_embed(templates_embedding_file)
        
        self.logger.info(f"All data preparation finished in {time.time() - start_time:.2f}s")

    def _train_parser(self, parser: Drain3Parser) -> None:
        """Train the Drain parser on log data.
        
        Args:
            parser: Initialized Drain3Parser instance
        """
        self._ensure_logger()
        self.logger.info("No trained parser found, start training")
        
        if hasattr(self, "ab_in_file") and self.ab_in_file:
            parser.parse_file_os(self.in_file, self.ab_in_file, remove_cols=self.remove_cols)
        else:
            parser.parse_file(self.in_file, remove_cols=self.remove_cols)
            
        self.logger.info(f"Found {len(parser.parser.drain.clusters)} templates")

    def _load_templates_from_parser(self, parser: Drain3Parser) -> None:
        """Load templates from trained parser.
        
        Args:
            parser: Trained Drain3Parser instance
        """
        for cluster in parser.parser.drain.clusters:
            self.templates[int(cluster.cluster_id)] = cluster.get_template()

    def _generate_parsing_results(
        self, 
        parser: Drain3Parser,
        log_template_mapping_file: Path,
        log_event_seq_file: Path,
        core_jobs: int
    ) -> None:
        """Generate parsing results from scratch.
        
        Args:
            parser: Trained Drain3Parser instance
            log_template_mapping_file: Output file for log-to-template mapping
            log_event_seq_file: Output file for log event sequences
            core_jobs: Number of parallel jobs for parsing
        """
        self._ensure_logger()
        self.logger.info("Missing persistence file(s), starting full parsing process")
        self.logger.warning(
            f"If you don't want this to happen, please copy persistence files to {parser.persistence_folder}"
        )
        
        # Prepare log lines for parsing
        ori_lines = self._prepare_log_lines()
        
        # Perform parsing
        self.logger.info("Parsing raw logs...")
        start_time = time.time()
        
        if core_jobs:
            self._parallel_parsing(parser, ori_lines, core_jobs)
        else:
            self._sequential_parsing(parser, ori_lines)
            
        self.logger.info(f"Finished parsing in {time.time() - start_time:.2f}s")
        
        # Create event sequences from log IDs
        self._create_event_sequences()
        
        # Save results to files
        self._record_parsing_results(log_template_mapping_file, log_event_seq_file)

    def _prepare_log_lines(self) -> List[Tuple[int, str]]:
        """Prepare log lines for parsing.
        
        Returns:
            List of (log_id, processed_line) tuples
        """
        ori_lines = []
        log_id = 0
        
        # Process main input file
        if self.in_file:
            with open(self.in_file, "r", encoding="utf-8") as reader:
                for line in tqdm(reader.readlines(), desc="Processing main log file"):
                    processed_line = self._pre_process(line)
                    ori_lines.append((log_id, processed_line))
                    log_id += 1
        
        # Process additional file if available
        if hasattr(self, "ab_in_file") and self.ab_in_file:
            with open(self.ab_in_file, "r", encoding="utf-8") as reader:
                for line in tqdm(reader.readlines(), desc="Processing additional log file"):
                    processed_line = self._pre_process(line)
                    ori_lines.append((log_id, processed_line))
                    log_id += 1
                    
        return ori_lines

    def _parallel_parsing(self, parser: Drain3Parser, lines: List[Tuple[int, str]], core_jobs: int) -> None:
        """Parse log lines in parallel.
        
        Args:
            parser: Trained Drain3Parser instance
            lines: List of (log_id, log_line) tuples to parse
            core_jobs: Number of parallel jobs
        """
        m = Manager()
        log2temp = m.dict()
        pool = Pool(core_jobs)
        
        # Split lines among workers
        split_lines = self._split(lines, core_jobs)
        inputs = zip([parser] * core_jobs, split_lines, [log2temp] * core_jobs)
        
        # Run parsing in parallel
        pool.starmap(_async_parsing, inputs)
        pool.close()
        pool.join()
        
        # Store results
        self.log2temp = dict(log2temp)

    def _sequential_parsing(self, parser: Drain3Parser, lines: List[Tuple[int, str]]) -> None:
        """Parse log lines sequentially.
        
        Args:
            parser: Trained Drain3Parser instance
            lines: List of (log_id, log_line) tuples to parse
        """
        for log_id, line in tqdm(lines, desc="Parsing logs"):
            cluster = parser.match(line)
            self.log2temp[log_id] = cluster.cluster_id

    def _create_event_sequences(self) -> None:
        """Create event sequences from log IDs."""
        for block, seq in self.block2seqs.items():
            self.block2eventseq[block] = [self.log2temp[log_id] for log_id in seq]

    def load_parsing_results(self, log_template_mapping_file: Union[str, Path], event_seq_file: Union[str, Path]) -> None:
        """Load parsing results from files.
        
        Args:
            log_template_mapping_file: File containing log-to-template mapping
            event_seq_file: File containing log event sequences
        """
        self._ensure_logger()
        self.logger.info("Loading previous parsing results")
        start = time.time()
        
        with open(log_template_mapping_file, "r", encoding="utf-8") as mapping_reader:
            self._load_log2temp(mapping_reader)
            
        with open(event_seq_file, "r", encoding="utf-8") as seq_reader:
            self._load_log_event_seqs(seq_reader)
            
        self.logger.info(f"Finished loading in {time.time() - start:.2f}s")

    def _restore(self) -> None:
        """Reset parsing state."""
        self.block2emb = {}
        self.templates = {}
        self.log2temp = {}

    def _record_parsing_results(self, log_template_mapping_file: Path, event_seq_file: Path) -> None:
        """Record parsing results to files.
        
        Args:
            log_template_mapping_file: Output file for log-to-template mapping
            event_seq_file: Output file for log event sequences
        """
        self._ensure_logger()
        start_time = time.time()
        
        with open(log_template_mapping_file, "w", encoding="utf-8") as mapping_writer:
            self._save_log2temp(mapping_writer)
            
        with open(event_seq_file, "w", encoding="utf-8") as seq_writer:
            self._save_log_event_seqs(seq_writer)
            
        self.logger.info(f"Results saved in {time.time() - start_time:.2f}s")

    def _save_log_event_seqs(self, writer: Any) -> None:
        """Save log event sequences to a file.
        
        Args:
            writer: File-like object for writing
        """
        self._ensure_logger()
        self.logger.info("Saving log event sequences")
        
        for block, event_seq in self.block2eventseq.items():
            seq_str = " ".join(map(str, event_seq))
            writer.write(f"{block}:{seq_str}\n")
            
        self.logger.info("Log event sequences saved")

    def _load_log_event_seqs(self, reader: Any) -> None:
        """Load log event sequences from a file.
        
        Args:
            reader: File-like object for reading
        """
        for line in reader.readlines():
            tokens = line.strip().split(":")
            block = tokens[0]
            seq = tokens[1].split()
            self.block2eventseq[block] = [int(x) for x in seq]
            
        self._ensure_logger()
        self.logger.info(f"Loaded {len(self.block2eventseq)} blocks")

    def _prepare_semantic_embed(self, semantic_emb_file: Path) -> None:
        """Prepare semantic embeddings for templates.
        
        Args:
            semantic_emb_file: Output file for semantic embeddings
        """
        self._ensure_logger()
        
        if self.semantic_repr_func:
            # Generate embeddings
            self.id2embed = self.semantic_repr_func(self.templates)
            
            # Save embeddings to file
            with open(semantic_emb_file, "w", encoding="utf-8") as writer:
                for template_id, embed in self.id2embed.items():
                    embed_str = " ".join(str(x) for x in embed.tolist())
                    writer.write(f"{template_id} {embed_str}\n")
                    
            self.logger.info(f"Semantic representations saved to {semantic_emb_file}")
        else:
            self.logger.warning(
                "No template encoder. This may lead to duplicate full parsing process."
            )

    def _check_parsing_persistences(self, log_template_mapping_file: Path, event_seq_file: Path) -> bool:
        """Check if parsing persistence files exist and are valid.
        
        Args:
            log_template_mapping_file: File containing log-to-template mapping
            event_seq_file: File containing log event sequences
            
        Returns:
            True if both files exist and are not empty
        """
        mapping_valid = self._check_file_validity(log_template_mapping_file)
        seq_valid = self._check_file_validity(event_seq_file)
        return mapping_valid and seq_valid

    def _check_file_validity(self, file_path: Path) -> bool:
        """Check if a file exists and is not empty.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file exists and is not empty
        """
        valid = file_path.exists() and file_path.stat().st_size > 0
        self._ensure_logger()
        self.logger.info(f"Checking file {file_path} ... {valid}")
        return valid

    def _load_templates(self, reader: Any) -> None:
        """Load templates from a file.
        
        Args:
            reader: File-like object for reading
        """
        for line in reader.readlines():
            tokens = line.strip().split(",")
            template_id = int(tokens[0])
            template = ",".join(tokens[1:])
            self.templates[template_id] = template
            
        self._ensure_logger()
        self.logger.info(f"Loaded {len(self.templates)} templates")

    def _save_templates(self, writer: Any) -> None:
        """Save templates to a file.
        
        Args:
            writer: File-like object for writing
        """
        for template_id, template in self.templates.items():
            writer.write(f"{template_id},{template}\n")
            
        self._ensure_logger()
        self.logger.info("Templates saved")

    def _load_log2temp(self, reader: Any) -> None:
        """Load log-to-template mapping from a file.
        
        Args:
            reader: File-like object for reading
        """
        for line in reader.readlines():
            logid, tempid = line.strip().split(",")
            self.log2temp[int(logid)] = int(tempid)
            
        self._ensure_logger()
        self.logger.info(f"Loaded {len(self.log2temp)} log mappings")

    def _save_log2temp(self, writer: Any) -> None:
        """Save log-to-template mapping to a file.
        
        Args:
            writer: File-like object for writing
        """
        for log_id, temp_id in self.log2temp.items():
            writer.write(f"{log_id},{temp_id}\n")
            
        self._ensure_logger()
        self.logger.info("Log mappings saved")

    def _load_semantic_embed(self, reader: Any) -> None:
        """Load semantic embeddings from a file.
        
        Args:
            reader: File-like object for reading
        """
        for line in reader.readlines():
            tokens = line.split()
            template_id = int(tokens[0])
            embed = np.asarray(tokens[1:], dtype=float)
            self.id2embed[template_id] = embed
            
        self._ensure_logger()
        dim = next(iter(self.id2embed.values())).shape[0] if self.id2embed else 0
        self.logger.info(f"Loaded {len(self.id2embed)} embeddings with dimension {dim}")

    def _split(self, items: List[Any], copies: int = 5) -> List[List[Any]]:
        """Split a list into approximately equal parts.
        
        Args:
            items: List to split
            copies: Number of parts
            
        Returns:
            List of sub-lists
        """
        quota = len(items) // copies + 1
        return [items[i * quota:(i + 1) * quota] for i in range(copies)]
