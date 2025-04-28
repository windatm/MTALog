"""Unified preprocessing module for log datasets with parsing and semantic representation."""

import gc
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

from tqdm import tqdm

from constants import LOG_ROOT, PROJECT_ROOT, SESSION
from core.entities.instances import Instance
from preprocessing.dataloader.BGLLoader import BGLLoader
from preprocessing.dataloader.HDFSLoader import HDFSLoader
from preprocessing.dataloader.OpenStackLoader import OpenStackLoader


class DatasetType(Enum):
    """Supported dataset types."""
    HDFS = auto()
    BGL = auto()
    BGL_SAMPLE = auto()
    OPENSTACK = auto()
    
    @classmethod
    def from_string(cls, name: str) -> "DatasetType":
        """Convert string dataset name to enum.
        
        Args:
            name: Dataset name string
            
        Returns:
            Corresponding DatasetType enum
            
        Raises:
            ValueError: If dataset name is not recognized
        """
        name_lower = name.lower()
        if name_lower == "hdfs":
            return cls.HDFS
        elif name_lower == "bgl":
            return cls.BGL
        elif name_lower == "bglsample":
            return cls.BGL_SAMPLE
        elif name_lower == "openstack":
            return cls.OPENSTACK
        else:
            raise ValueError(f"Unknown dataset: {name}")


class ParsingMethod(Enum):
    """Supported log parsing methods."""
    IBM = auto()
    
    @classmethod
    def from_string(cls, name: str) -> "ParsingMethod":
        """Convert string parser name to enum.
        
        Args:
            name: Parser name string
            
        Returns:
            Corresponding ParsingMethod enum
            
        Raises:
            ValueError: If parser name is not recognized
        """
        name_lower = name.lower()
        if name_lower == "ibm":
            return cls.IBM
        else:
            raise ValueError(f"Unknown parsing method: {name}")


@dataclass
class DataSplit:
    """Container for dataset splits."""
    train: List[Instance] = field(default_factory=list)
    val: List[Instance] = field(default_factory=list)
    test: List[Instance] = field(default_factory=list)


# Type for template encoding function
TemplateEncoder = Callable[[Dict[int, str]], Dict[int, Any]]

# Type for dataset cutting function 
CutFunc = Callable[[List[Instance]], Tuple[List[Instance], List[Instance], List[Instance]]]


class Preprocessor:
    """Unified preprocessing interface for log anomaly detection datasets.
    
    Transforms raw logs into structured instances with template parsing and semantic encoding.
    Handles dataset loading, parsing, encoding, and train/val/test splitting.
    
    Supported datasets: HDFS, BGL, BGLSample, OpenStack  
    Supported parsing: IBM (Drain-based log template parser)
    
    Attributes:
        templates: Parsed log templates
        embedding: Event-to-vector mappings
        train_event2idx: Event-index mapping for training data
        test_event2idx: Event-index mapping extended to include test events
        id2label: Mapping from label IDs to string labels
        label2id: Mapping from string labels to label IDs
    """
    
    def __init__(self) -> None:
        """Initialize the Preprocessor with empty state."""
        # Initialize logger
        self.logger = self._setup_logger()
        
        # DataLoader
        self.dataloader = None
        
        # Data mappings
        self.train_event2idx: Dict[int, int] = {}
        self.test_event2idx: Dict[int, int] = {}
        self.id2label: Dict[int, str] = {}
        self.label2id: Dict[str, int] = {}
        
        # Templates and embeddings
        self.templates: Dict[int, str] = {}
        self.embedding: Dict[int, Any] = {}
        
        # Dataset info
        self.base_path: Optional[Path] = None
        self.dataset_type: Optional[DatasetType] = None
        self.parsing_method: Optional[ParsingMethod] = None
        
        # Default label mappings
        self.tag2id: Dict[str, int] = {"Normal": 0, "Anomalous": 1}
        self.id2tag: Dict[int, str] = {0: "Normal", 1: "Anomalous"}

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Set up and configure logger.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger("Preprocessor")
        logger.setLevel(logging.DEBUG)
        
        # Create formatters and handlers
        log_format = f"%(asctime)s - %(name)s - {SESSION} - %(levelname)s: %(message)s"
        formatter = logging.Formatter(log_format)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        
        # File handler
        log_file = Path(LOG_ROOT) / "Preprocessor.log"
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Add handlers if they don't exist
        if not logger.handlers:
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
            
        logger.info(
            f"Logger initialized, CWD: {os.getcwd()}, logs: {LOG_ROOT}"
        )
        
        return logger

    def process(
        self, 
        dataset: str, 
        parsing: str, 
        template_encoding: TemplateEncoder,
        cut_func: CutFunc, 
        target_mode: bool = False
    ) -> Tuple[List[Instance], List[Instance], List[Instance]]:
        """Process a dataset with specified parsing and encoding.
        
        Main pipeline that handles:
        1. Loading the appropriate dataset
        2. Parsing logs into templates
        3. Encoding templates into vectors
        4. Splitting data into train/val/test sets
        
        Args:
            dataset: Dataset name ('HDFS', 'BGL', 'BGLSample', 'OpenStack')
            parsing: Parsing method ('IBM')
            template_encoding: Function to encode templates to vectors
            cut_func: Function to split instances into train/val/test
            target_mode: Whether to filter embeddings to only include train IDs
            
        Returns:
            Tuple of (train, val, test) instance lists
            
        Raises:
            ValueError: For unknown dataset or parsing method
        """
        # Convert string parameters to enums
        self.dataset_type = DatasetType.from_string(dataset)
        self.parsing_method = ParsingMethod.from_string(parsing)
        
        # Set up base path
        self.base_path = Path(PROJECT_ROOT) / f"datasets/{dataset}/inputs/{parsing}"
        
        # Initialize appropriate dataloader
        self.dataloader = self._create_dataloader(
            self.dataset_type, template_encoding
        )
        
        # Get parser configuration
        parser_config = self._get_parser_config(self.dataset_type)
        parser_persistence = Path(PROJECT_ROOT) / f"datasets/{dataset}/persistences"
        
        # Parse according to selected method
        if self.parsing_method == ParsingMethod.IBM:
            self.dataloader.parse_by_IBM(
                config_file=parser_config,
                persistence_folder=parser_persistence
            )
        else:
            raise NotImplementedError(f"Parsing method {parsing} not implemented")
            
        # Generate instances and apply splits
        return self._generate_instances(cut_func, target_mode)

    def _create_dataloader(
        self, 
        dataset_type: DatasetType, 
        template_encoding: TemplateEncoder
    ) -> Any:
        """Create appropriate dataloader for the selected dataset.
        
        Args:
            dataset_type: Type of dataset to load
            template_encoding: Function to encode templates
            
        Returns:
            Initialized dataloader instance
            
        Raises:
            ValueError: For unknown dataset types
        """
        project_root = Path(PROJECT_ROOT)
        
        if dataset_type == DatasetType.HDFS:
            return HDFSLoader(
                in_file=project_root / "datasets/HDFS/HDFS.log",
                semantic_repr_func=template_encoding
            )
            
        elif dataset_type in (DatasetType.BGL, DatasetType.BGL_SAMPLE):
            dataset_name = "BGL" if dataset_type == DatasetType.BGL else "BGLSample"
            dataset_path = project_root / f"datasets/{dataset_name}"
            
            return BGLLoader(
                in_file=dataset_path / f"{dataset_name}.log",
                dataset_base=dataset_path,
                semantic_repr_func=template_encoding
            )
            
        elif dataset_type == DatasetType.OPENSTACK:
            dataset_path = project_root / "datasets/OpenStack"
            
            return OpenStackLoader(
                in_file=dataset_path / "openstack_normal1.log",
                ab_in_file=dataset_path / "openstack_abnormal.log",
                dataset_base=dataset_path,
                semantic_repr_func=template_encoding
            )
            
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def _get_parser_config(self, dataset_type: DatasetType) -> Path:
        """Get parser configuration file path for dataset.
        
        Args:
            dataset_type: Type of dataset
            
        Returns:
            Path to appropriate config file
        """
        config_dir = Path(PROJECT_ROOT) / "conf"
        
        if dataset_type == DatasetType.HDFS:
            return config_dir / "HDFS.ini"
        elif dataset_type in (DatasetType.BGL, DatasetType.BGL_SAMPLE):
            return config_dir / "BGL.ini"
        elif dataset_type == DatasetType.OPENSTACK:
            return config_dir / "OpenStack.ini"
        else:
            raise ValueError(f"No config for dataset: {dataset_type}")

    def _generate_instances(
        self, 
        cut_func: CutFunc, 
        target_mode: bool = False
    ) -> Tuple[List[Instance], List[Instance], List[Instance]]:
        """Generate structured instances from parsed logs and split into sets.
        
        Args:
            cut_func: Function to split instances into train/val/test
            target_mode: Whether to filter embeddings to only include train IDs
            
        Returns:
            Tuple of (train, val, test) instance lists
        """
        self.logger.info(
            f"Preprocessing dataset {self.dataset_type.name} with {self.parsing_method.name}"
        )
        
        # Ensure output directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Output file paths
        train_file = self.base_path / "train"
        val_file = self.base_path / "val"
        test_file = self.base_path / "test"
        
        # Generate instances from parsed logs
        self.logger.info("Generating instances from parsed logs")
        instances = self._create_instances_from_blocks()
        
        # Apply splitting function
        train, val, test = cut_func(instances)
        
        # Filter embeddings if in target mode
        if target_mode:
            train_ids = {int(event_id) for inst in train for event_id in inst.sequence}
            self.embedding = {
                eid: emb for eid, emb in self.dataloader.id2embed.items() 
                if eid in train_ids
            }
        else:
            self.embedding = self.dataloader.id2embed
        
        # Log label distribution
        self._log_label_distribution(train, val, test)
        
        # Save splits to files
        self._save_splits(train, val, test, train_file, val_file, test_file)
        
        # Update internal dictionaries
        self._update_dictionaries()
        self._update_event_mappings(train, test)
        
        # Cleanup
        del self.dataloader
        gc.collect()
        
        return train, val, test

    def _create_instances_from_blocks(self) -> List[Instance]:
        """Create Instance objects from parsed log blocks.
        
        Returns:
            List of Instance objects
        """
        instances = []
        
        for block in tqdm(self.dataloader.blocks, desc="Creating instances"):
            if (block in self.dataloader.block2eventseq and 
                block in self.dataloader.block2label):
                
                block_id = block
                label = self.dataloader.block2label[block_id]
                event_seq = self.dataloader.block2eventseq[block_id]
                
                instance = Instance(block_id, event_seq, label)
                instances.append(instance)
            else:
                self.logger.error(f"Mismatched block: {block}")
                
        return instances

    def _update_dictionaries(self) -> None:
        """Update internal dictionaries from dataloader."""
        self.id2label = self.dataloader.id2label
        self.label2id = self.dataloader.label2id
        self.templates = self.dataloader.templates

    def _save_splits(
        self,
        train: List[Instance],
        val: List[Instance],
        test: List[Instance],
        train_file: Path,
        val_file: Path,
        test_file: Path,
        pretrain_source: Optional[Path] = None
    ) -> None:
        """Save split instances to files.
        
        Args:
            train: Training instances
            val: Validation instances
            test: Test instances
            train_file: Path to save train instances
            val_file: Path to save validation instances
            test_file: Path to save test instances
            pretrain_source: Optional path to save pretraining data
        """
        # Save train instances
        with open(train_file, "w", encoding="utf-8") as f:
            for instance in train:
                f.write(f"{instance}\n")
        
        # Save validation instances if any
        if val:
            with open(val_file, "w", encoding="utf-8") as f:
                for instance in val:
                    f.write(f"{instance}\n")
        
        # Save test instances
        with open(test_file, "w", encoding="utf-8") as f:
            for instance in test:
                f.write(f"{instance}\n")
        
        # Save pretraining data if requested
        if pretrain_source:
            with open(pretrain_source, "w", encoding="utf-8") as f:
                for instance in train:
                    sequence_str = " ".join(str(x) for x in instance.sequence)
                    f.write(f"{sequence_str}\n")

    def _log_label_distribution(
        self, 
        train: List[Instance], 
        val: List[Instance], 
        test: List[Instance]
    ) -> None:
        """Log the distribution of labels in each data split.
        
        Args:
            train: Training instances
            val: Validation instances
            test: Test instances
        """
        train_labels = Counter(inst.label for inst in train)
        
        self.logger.info(
            f"Train: {train_labels['Normal']} Normal, {train_labels['Anomalous']} Anomalous"
        )
        
        if val:
            val_labels = Counter(inst.label for inst in val)
            self.logger.info(
                f"Val: {val_labels['Normal']} Normal, {val_labels['Anomalous']} Anomalous"
            )
            
        test_labels = Counter(inst.label for inst in test)
        self.logger.info(
            f"Test: {test_labels['Normal']} Normal, {test_labels['Anomalous']} Anomalous"
        )

    def _update_event_mappings(
        self, 
        train_instances: List[Instance], 
        test_instances: List[Instance]
    ) -> None:
        """Update event-to-index mappings for train and test sets.
        
        Creates consistent indexing for events across train and test,
        with new events in test getting indices after all train events.
        
        Args:
            train_instances: Training instances
            test_instances: Test instances
        """
        self.logger.info("Updating event-to-index mappings")
        
        # Get ordered events from training set
        train_events = self._collect_unique_events(train_instances)
        train_event_count = len(train_events)
        
        self.logger.info(f"Training set contains {train_event_count} unique events")
        
        # Create train mapping (events -> indices)
        self.train_event2idx = {event: idx for idx, event in enumerate(train_events)}
        
        # Get ordered events from test set
        test_events = self._collect_unique_events(test_instances)
        
        # Create test mapping, adding new events with indices after train events
        self.test_event2idx = {}
        next_idx = train_event_count
        
        for event in test_events:
            if event in self.train_event2idx:
                # Reuse index for events seen in training
                self.test_event2idx[event] = self.train_event2idx[event]
            else:
                # Assign new index for previously unseen events
                self.test_event2idx[event] = next_idx
                next_idx += 1
                
        # Add training events not in test set to complete the mapping
        for event in train_events:
            if event not in self.test_event2idx:
                self.test_event2idx[event] = self.train_event2idx[event]
                
        total_events = next_idx
        self.logger.info(f"Combined set contains {total_events} unique events")

    def _collect_unique_events(self, instances: List[Instance]) -> List[int]:
        """Collect and sort unique event IDs from instances.
        
        Args:
            instances: List of instances to extract events from
            
        Returns:
            Sorted list of unique event IDs
        """
        events = set()
        for instance in instances:
            events.update(int(event) for event in instance.sequence)
        return sorted(events)
