import gc
import logging
import os
import sys
from collections import Counter

from tqdm import tqdm

from CONSTANTS import LOG_ROOT, PROJECT_ROOT, SESSION
from entities.instances import Instance
from preprocessing.dataloader.BGLLoader import BGLLoader
from preprocessing.dataloader.HDFSLoader import HDFSLoader
from preprocessing.dataloader.OSLoader import OSLoader


class Preprocessor:
    """
    A unified preprocessing interface for loading log datasets, parsing raw logs into templates,
    generating semantic representations, and splitting data for anomaly detection experiments.

    Datasets supported: HDFS, BGL, BGLSample  
    Parsing supported: IBM (Drain-based log template parser)

    Responsibilities:
        - Load raw log data and labels
        - Parse log lines into structured event templates
        - Apply semantic encoding (via template_embedding function)
        - Split dataset into train/dev/test using a custom strategy
        - Write processed data to disk and update internal dictionaries

    Attributes:
        dataloader (BaseLoader): Dataset-specific dataloader (e.g., HDFSLoader, BGLLoader).
        templates (list): Parsed log templates.
        embedding (dict): Event-to-vector mappings.
        train_event2idx (dict): Event-index mapping for training data.
        test_event2idx (dict): Event-index mapping extended to include test events.
        tag2id (dict): Label-to-ID mapping (e.g., {"Normal": 0, "Anomalous": 1}).
        id2tag (dict): ID-to-Label mapping.
    """
    _logger = logging.getLogger("Preprocessor")
    _logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
        )
    )
    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, "Preprocessor.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
        )
    )
    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    _logger.info(
        f"Construct logger for MTALog succeeded, current working directory: {os.getcwd()}, logs will be written in {LOG_ROOT}"
    )

    @property
    def logger(self):
        return Preprocessor._logger

    def __init__(self):
        """
        Initialize the Preprocessor instance. Sets up internal attributes, label dictionaries,
        and logging infrastructure.
        """
        self.dataloader = None
        self.train_event2idx = {}
        self.test_event2idx = {}
        self.id2label = {}
        self.label2id = {}
        self.templates = []
        self.embedding = None
        self.base = None
        self.dataset = None
        self.parsing = None
        self.tag2id = {"Normal": 0, "Anomalous": 1}
        self.id2tag = {0: "Normal", 1: "Anomalous"}

    def process(self, dataset, parsing, template_encoding, cut_func):
        """
        Main preprocessing pipeline for a given dataset and parser.

        Args:
            dataset (str): Dataset name (e.g., 'HDFS', 'BGL', 'BGLSample').
            parsing (str): Log parsing method (currently supports only 'IBM' → Drain).
            template_encoding (function): Semantic encoder function that maps event templates to vectors.
            cut_func (function): A callable to split the instance list into train/dev/test sets.

        Returns:
            tuple: (train, dev, test), each a list of Instance objects.
        """

        self.base = os.path.join(
            PROJECT_ROOT, "datasets/" + dataset + "/inputs/" + parsing
        )
        self.dataset = dataset
        self.parsing = parsing
        dataloader = None
        parser_config = None
        parser_persistence = os.path.join(
            PROJECT_ROOT, "datasets/" + dataset + "/persistences"
        )

        if dataset == "HDFS":
            dataloader = HDFSLoader(
                in_file=os.path.join(PROJECT_ROOT, "datasets/HDFS/HDFS.log"),
                semantic_repr_func=template_encoding,
            )
            parser_config = os.path.join(PROJECT_ROOT, "conf/HDFS.ini")
        elif dataset == "BGL" or dataset == "BGLSample":
            in_file = os.path.join(
                PROJECT_ROOT, "datasets/" + dataset + "/" + dataset + ".log"
            )
            dataset_base = os.path.join(PROJECT_ROOT, "datasets/" + dataset)
            dataloader = BGLLoader(
                in_file=in_file,
                dataset_base=dataset_base,
                semantic_repr_func=template_encoding,
            )
            parser_config = os.path.join(PROJECT_ROOT, "conf/BGL.ini")

        self.dataloader = dataloader

        if parsing == "IBM":
            self.dataloader.parse_by_IBM(
                config_file=parser_config, persistence_folder=parser_persistence
            )
        else:
            self.logger.error("Parsing method %s not implemented yet.")
            raise NotImplementedError
        return self._gen_instances(cut_func=cut_func)

    def _gen_instances(self, cut_func=None):
        """
        Internal function to convert raw parsed log blocks into Instance objects, apply cut strategy,
        and save intermediate results.

        Args:
            cut_func (callable): Function to split the data into train/dev/test.

        Returns:
            tuple: (train, dev, test) lists of Instance objects.
        """
        self.logger.info(
            "Start preprocessing dataset %s by parsing method %s"
            % (self.dataset, self.parsing)
        )
        instances = []
        if not os.path.exists(self.base):
            os.makedirs(self.base)
        train_file = os.path.join(self.base, "train")
        dev_file = os.path.join(self.base, "dev")
        test_file = os.path.join(self.base, "test")

        self.logger.info("Start generating instances.")
        # Prepare semantic embedding sequences for instances.
        for block in tqdm(self.dataloader.blocks):
            if (
                block in self.dataloader.block2eventseq.keys()
                and block in self.dataloader.block2label.keys()
            ):
                id = block
                label = self.dataloader.block2label[id]
                inst = Instance(id, self.dataloader.block2eventseq[id], label)
                instances.append(inst)
            else:
                self.logger.error("Found mismatch block: %s. Please check." % block)
        self.embedding = self.dataloader.id2embed

        train, dev, test = cut_func(instances)
        self.label_distribution(train, dev, test)
        self.record_files(train, train_file, dev, dev_file, test, test_file)
        self.update_dicts()
        self.update_event2idx_mapping(train, test)
        del self.dataloader
        gc.collect()
        return train, dev, test

    def update_dicts(self):
        """
        Update internal template and label dictionaries using the dataloader outputs.
        """
        self.id2label = self.dataloader.id2label
        self.label2id = self.dataloader.label2id
        self.templates = self.dataloader.templates

    def record_files(
        self, train, train_file, dev, dev_file, test, test_file, pretrain_source=None
    ):
        """
        Write processed train/dev/test instances to disk as `.txt` files.

        Args:
            train (list): Training instances.
            train_file (str): Path to write training data.
            dev (list): Dev instances (can be None).
            dev_file (str): Path to write dev data.
            test (list): Test instances.
            test_file (str): Path to write test data.
            pretrain_source (str, optional): If specified, also saves only token sequences for pretraining.
        """
        with open(train_file, "w", encoding="utf-8") as writer:
            for instance in train:
                writer.write(str(instance) + "\n")
        if dev:
            with open(dev_file, "w", encoding="utf-8") as writer:
                for instance in dev:
                    writer.write(str(instance) + "\n")
        with open(test_file, "w", encoding="utf-8") as writer:
            for instance in test:
                writer.write(str(instance) + "\n")
        if pretrain_source:
            with open(pretrain_source, "w", encoding="utf-8") as writer:
                for inst in train:
                    writer.write(" ".join([str(x) for x in inst.sequence]) + "\n")

    def label_distribution(self, train, dev, test):
        """
        Logs the distribution of labels (Normal / Anomalous) in each dataset split.

        Args:
            train (list): Training set.
            dev (list): Dev set.
            test (list): Test set.
        """
        train_label_counter = Counter([inst.label for inst in train])
        if dev:
            dev_label_counter = Counter([inst.label for inst in dev])
            self.logger.info(
                "Dev: %d Normal, %d Anomalous instances.",
                dev_label_counter["Normal"],
                dev_label_counter["Anomalous"],
            )
        test_label_counter = Counter([inst.label for inst in test])
        self.logger.info(
            "Train: %d Normal, %d Anomalous instances.",
            train_label_counter["Normal"],
            train_label_counter["Anomalous"],
        )
        self.logger.info(
            "Test: %d Normal, %d Anomalous instances.",
            test_label_counter["Normal"],
            test_label_counter["Anomalous"],
        )

    def update_event2idx_mapping(self, pre, post):
        """
        Build index mappings for unique event IDs in training and test data for use in event count vectors.

        Args:
            pre (list): Pre-training instances (usually train + dev).
            post (list): Post-training instances (usually test).
        """
        self.logger.info("Update train instances' event-idx mapping.")
        pre_ordered_events = self._count_events(pre)
        embed_size = len(pre_ordered_events)
        self.logger.info("Embed size: %d in pre dataset." % embed_size)
        for idx, event in enumerate(pre_ordered_events):
            self.train_event2idx[event] = idx
        self.logger.info("Update test instances' event-idx mapping.")
        post_ordered_events = self._count_events(post)
        base = len(pre_ordered_events)
        increment = 0
        for event in post_ordered_events:
            if event not in pre_ordered_events:
                pre_ordered_events.append(event)
                self.test_event2idx[event] = base + increment
                increment += 1
            else:
                self.test_event2idx[event] = self.train_event2idx[event]
        embed_size = len(pre_ordered_events)
        self.logger.info("Embed size: %d in pre+post dataset." % embed_size)
        pass

    def _count_events(self, sequence):
        """
        Count unique event IDs from a sequence of instances.

        Args:
            sequence (list): List of Instance objects.

        Returns:
            list: Sorted list of unique integer event IDs.
        """
        events = set()
        for inst in sequence:
            for event in inst.sequence:
                events.add(int(event))
        ordered_events = sorted(list(events))
        return ordered_events
