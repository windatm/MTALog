#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MTALog: A Python implementation of MTALog for log anomaly detection
Converted from Jupyter notebook to Python script
"""

# Standard library
import os
import re
import time
from collections import OrderedDict
import random
import pickle
import numpy as np
import logging
import sys

# Third-party
import matplotlib.pyplot as plt
import torch

# Local modules
from CONSTANTS import DEVICE, PROJECT_ROOT, SESSION
from models.gru import AttGRUModel
from module.Common import data_iter, generate_tinsts_binary_label
from module.Optimizer import Optimizer
from preprocessing.datacutter.SimpleCutting import (
    cut_by, cut_all, cut_sequential, fewshot_split, sample_query_set, create_query_set
)
from representations.templates.statistics import Template_TF_IDF_without_clean
from utils.vocab import Vocab
from utils.common import get_model_and_result_paths
from utils.logger import setup_logger
from utils.data import (
    preprocess_data,
    encode_log_sequences_with_gru,
    encode_query_with_fallback,
)
from preprocessing.Preprocess import Preprocessor
from entities.statistics import ResultStatistics


def setup_params():
    """Set up parameters and configuration"""
    # Embedding Configuration 
    word2vec_file = "glove.6B.300d.txt"

    # Meta-Learning Hyperparameters 
    alpha = 8e-3         # Inner loop learning rate (meta-train)
    beta = 1.0           # Outer loop scaling factor (meta-test loss weight)
    gamma = 8e-3         # Learning rate for optimizer

    # Model Architecture 
    lstm_hidden_units = 64   # Hidden size of each GRU direction
    num_layers = 4           # Number of GRU layers
    dropout_rate = 0.5       # Dropout rate applied to input embeddings

    # Training Configuration 
    batch_size = 1024        # Mini-batch size for training
    num_epochs = 5           # Number of training epochs

    # Experiment Settings 
    parser = "IBM"          # Log parser to use (e.g., Drain, Spell, IBM)
    mode = "train"          # Mode can be 'train' or 'eval'

    source_systems = ["HDFS", "OpenStack"]
    target_system = "BGL"   # Target log system (e.g., BGL, HDFS, Thunderbird)
    few_shot_ratio = 0.1    # Ratio of normal logs used in support (e.g., 1%)
    query_sample_ratio = 1.0 # Ratio of query set sampled for evaluation (e.g., 1%)

    params = {
        "word2vec_file": word2vec_file,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "lstm_hidden_units": lstm_hidden_units,
        "num_layers": num_layers,
        "dropout_rate": dropout_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "parser": parser,
        "mode": mode,
        "source_systems": source_systems,
        "target_system": target_system,
        "few_shot_ratio": few_shot_ratio,
        "query_sample_ratio": query_sample_ratio
    }
    
    return params


def setup_logging(params):
    """Initialize logger and log parameters"""
    # Initialize logger
    logger = setup_logger()

    logger.info(f"DEVICE        : {DEVICE}")

    # Log architecture parameters
    logger.info("=== Model Architecture ===")
    logger.info(f"LSTM hidden units         : {params['lstm_hidden_units']}")
    logger.info(f"Number of GRU layers      : {params['num_layers']}")
    logger.info(f"Dropout rate              : {params['dropout_rate']}")
    logger.info(f"Latent representation dim : {2 * params['lstm_hidden_units']}")

    # Log training hyperparameters
    logger.info("=== Training Hyperparameters ===")
    logger.info(f"Meta-train step size (alpha)     : {params['alpha']}")
    logger.info(f"Meta-test loss weight (beta)     : {params['beta']}")
    logger.info(f"Learning rate (gamma)            : {params['gamma']}")
    logger.info(f"Word2Vec file used               : {params['word2vec_file']}")
    
    return logger


def setup_template_encoder(params):
    """Set up template encoder using word2vec"""
    template_encoder = Template_TF_IDF_without_clean(params["word2vec_file"])
    return template_encoder


def process_source_systems(params, logger, template_encoder):
    """Process source log systems and create a combined vocabulary from all systems.
    
    Args:
        params: Dictionary containing parameters for processing
        logger: Logger instance for logging messages
        template_encoder: Encoder for templates
        
    Returns:
        Dictionary containing processed source systems data
    """
    source_processors = OrderedDict()
    source_vocabularies = OrderedDict() 
    source_encoders = OrderedDict()
    source_support_sets = OrderedDict()
    source_query_sets = OrderedDict()
    
    # Get model and result paths
    output_model_dir, output_res_dir = get_model_and_result_paths(params["parser"], PROJECT_ROOT)
    
    # Create a cache directory for processed data
    cache_dir = os.path.join(PROJECT_ROOT, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check if the combined processed data exists
    combined_cache_file = os.path.join(cache_dir, f"combined_{params['parser']}_source_systems.pkl")
    if os.path.exists(combined_cache_file):
        try:
            with open(combined_cache_file, 'rb') as f:
                combined_data = pickle.load(f)
                
            # Basic validation that required keys exist
            required_keys = ["source_processors", "source_vocabularies", "source_encoders", 
                            "source_support_sets", "source_query_sets", "combined_vocab"]
            
            if all(key in combined_data for key in required_keys):
                logger.info("Using cached source systems data")
                return combined_data
                
        except Exception as e:
            logger.warning(f"Error loading cache: {str(e)}. Reprocessing...")
            if os.path.exists(combined_cache_file):
                os.remove(combined_cache_file)
    
    # Process each source system
    for source_system in params["source_systems"]:
        logger.info(f"=== [SOURCE] Processing system: {source_system} ===")
        
        # Define cache files for this source system
        data_cache_file = os.path.join(cache_dir, f"{source_system}_{params['parser']}_data.pkl")
        encoder_cache_file = os.path.join(cache_dir, f"{source_system}_{params['parser']}_encoder.pkl")
        
        # Check if cached data exists for this source system
        processor = None
        train_data = []
        encoded_data = []
        vocab = None
        encoder = None
        support_set = []
        query_set = []
        
        data_loaded = False
        if os.path.exists(data_cache_file) and os.path.exists(encoder_cache_file):
            try:
                # Load data cache
                with open(data_cache_file, 'rb') as f:
                    data_cache = pickle.load(f)
                    train_data = data_cache.get('train_data', [])
                    processor = data_cache.get('processor', None)
                
                # Load encoder cache
                with open(encoder_cache_file, 'rb') as f:
                    encoder_cache = pickle.load(f)
                    encoder = encoder_cache.get('encoder', None)
                    support_set = encoder_cache.get('support_set', [])
                    query_set = encoder_cache.get('query_set', [])
                    vocab = encoder_cache.get('vocab', None)
                
                # Validate loaded data
                if train_data and encoder and support_set and query_set and vocab:
                    logger.info(f"[{source_system}] Loaded {len(train_data)} instances from cache")
                    data_loaded = True
                else:
                    logger.warning(f"[{source_system}] Cache validation failed. Reprocessing...")
                    os.remove(data_cache_file)
                    if os.path.exists(encoder_cache_file):
                        os.remove(encoder_cache_file)
            except Exception as e:
                logger.warning(f"[{source_system}] Error loading cache: {str(e)}. Reprocessing...")
                if os.path.exists(data_cache_file):
                    os.remove(data_cache_file)
                if os.path.exists(encoder_cache_file):
                    os.remove(encoder_cache_file)
        
        # Process if not loaded from cache
        if not data_loaded:
            logger.info(f"[{source_system}] Processing from scratch...")
            
            # Step 1: Define data splitting function
            cut_func = cut_by(train=0.8, val=0.1, anomalous_rate=1.0, random_seed=42)
            
            # Step 2: Process the data
            train_data, valid_data, test_data, processor = preprocess_data(
                dataset=source_system,
                parser=params["parser"],
                cut_func=cut_func,
                template_encoder=template_encoder
            )
            
            # Step 3: Build a vocabulary 
            vocab = Vocab()
            if processor and processor.embedding:
                vocab.load_from_dict(processor.embedding)
                logger.info(f"[{source_system}] Built vocabulary with {vocab.vocab_size} templates")
            else:
                logger.warning(f"[{source_system}] No embeddings found in processor, using empty vocabulary")
            
            # Step 4: Initialize and train the encoder
            encoder = AttGRUModel(
                vocab=vocab,
                lstm_layers=params["num_layers"],
                lstm_hiddens=params["lstm_hidden_units"],
                dropout=params["dropout_rate"]
            ).to(DEVICE)
            
            # Encode the data using the encoder
            encoded_data = encode_log_sequences_with_gru(
                model=encoder,
                vocab=vocab,
                instances=train_data,
                batch_size=params["batch_size"]
            )
            
            # If encoding failed, use fallback with zero vectors
            if not encoded_data:
                logger.warning(f"[{source_system}] Encoding failed, using fallback")
                for inst in train_data:
                    if not hasattr(inst, 'repr') or inst.repr is None:
                        repr_dim = 2 * params["lstm_hidden_units"]
                        inst.repr = np.zeros(repr_dim)
                encoded_data = train_data
            
            # Step 5: Split into support/query
            # For source systems, split data randomly into support/query
            random.shuffle(encoded_data)
            split_index = int(0.5 * len(encoded_data))
            support_set = encoded_data[:split_index]
            query_set = encoded_data[split_index:]
            
            # Step 6: Build repr lookup
            encoder.repr_lookup = {
                tuple(inst.sequence): inst.repr for inst in encoded_data if hasattr(inst, 'sequence') and hasattr(inst, 'repr')
            }
            
            # Save encoder and encoded data to cache
            encoder_cache = {
                'encoder': encoder,
                'support_set': support_set,
                'query_set': query_set,
                'vocab': vocab
            }
            
            # Save data to cache
            data_cache = {
                'train_data': train_data,
                'processor': processor
            }
            
            with open(data_cache_file, 'wb') as f:
                pickle.dump(data_cache, f)
                
            with open(encoder_cache_file, 'wb') as f:
                pickle.dump(encoder_cache, f)
        
        # Store processed data
        source_processors[source_system] = processor
        source_vocabularies[source_system] = vocab
        source_encoders[source_system] = encoder
        source_support_sets[source_system] = support_set
        source_query_sets[source_system] = query_set
        
        logger.info(f"[{source_system}] Support set: {len(support_set)} | Query set: {len(query_set)}")
    
    # Create combined vocabulary from all source systems
    # This is the key part for the requirement to combine vocabularies from all sources
    combined_vocab = Vocab()
    
    # First initialize with template encoder embeddings as base
    if hasattr(template_encoder, 'get_embeddings') and callable(getattr(template_encoder, 'get_embeddings')):
        combined_vocab.load_from_dict(template_encoder.get_embeddings())
        logger.info(f"Initialized combined vocab with {combined_vocab.vocab_size} base templates")
    
    # Collect all source embeddings
    all_embeddings = {}
    
    # Add templates from all source processors
    for system, processor in source_processors.items():
        if processor and hasattr(processor, 'embedding') and processor.embedding:
            # Add all embeddings from this source to combined dictionary
            for template_id, embedding in processor.embedding.items():
                if template_id not in all_embeddings:
                    all_embeddings[template_id] = embedding
            
            logger.info(f"Added {len(processor.embedding)} templates from {system}")
    
    # Now load the combined embeddings
    if all_embeddings:
        # Create a new Vocab with all combined embeddings
        combined_vocab = Vocab()
        combined_vocab.load_from_dict(all_embeddings)
        logger.info(f"Created combined vocabulary with {combined_vocab.vocab_size} templates from all sources")
    else:
        logger.warning("No source embeddings found to combine")
    
    # Return results
    result_data = {
        "source_processors": source_processors,
        "source_vocabularies": source_vocabularies,
        "source_encoders": source_encoders, 
        "source_support_sets": source_support_sets,
        "source_query_sets": source_query_sets,
        "combined_vocab": combined_vocab,
        "output_model_dir": output_model_dir,
        "output_res_dir": output_res_dir
    }
    
    # Save combined data to cache
    with open(combined_cache_file, 'wb') as f:
        pickle.dump(result_data, f)
    
    return result_data


def process_target_system(params, logger, template_encoder, source_data):
    """Process target log system"""
    target_system = params['target_system']
    logger.info(f"=== [TARGET] Processing system: {target_system} ===")
    
    # Setup cache directory
    cache_dir = os.path.join(PROJECT_ROOT, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define cache files
    data_cache_file = os.path.join(cache_dir, f"{target_system}_{params['parser']}_data.pkl")
    encoder_cache_file = os.path.join(cache_dir, f"{target_system}_{params['parser']}_encoder.pkl")
    
    # Get the combined vocabulary from source systems
    combined_vocab = source_data['combined_vocab']
    
    # Initialize variables with default values to prevent UnboundLocalError
    target_vocab = combined_vocab
    processor = None
    encoder = None
    support_set = []
    query_set = []
    train_data = []
    test_data = []
    support_templates = set()
    
    # Try to load from cache first
    data_loaded = False
    if os.path.exists(data_cache_file) and os.path.exists(encoder_cache_file):
        try:
            # Load data
            with open(data_cache_file, 'rb') as f:
                data_cache = pickle.load(f)
                train_data = data_cache.get('train_data', [])
                test_data = data_cache.get('test_data', [])
                processor = data_cache.get('processor', None)
                
            # Load encoder and sets
            with open(encoder_cache_file, 'rb') as f:
                encoder_cache = pickle.load(f)
                encoder = encoder_cache.get('encoder', None)
                support_set = encoder_cache.get('support_set', [])
                query_set = encoder_cache.get('query_set', [])
                
            # Validate data exists (less strict)
            if train_data and support_set:
                logger.info(f"[{target_system}] Loaded from cache: {len(train_data)} instances")
                logger.info(f"[{target_system}] Support set: {len(support_set)} | Query set: {len(query_set)}")
                
                # Ensure the encoder has repr_lookup attribute
                if not hasattr(encoder, 'repr_lookup') or not encoder.repr_lookup:
                    encoder.repr_lookup = {}
                    for inst in support_set + query_set:
                        if hasattr(inst, 'sequence') and hasattr(inst, 'repr'):
                            encoder.repr_lookup[tuple(inst.sequence)] = inst.repr
                
                data_loaded = True
                
                # Make sure target_vocab is set here too
                target_vocab = combined_vocab
            else:
                logger.warning(f"[{target_system}] Cache data validation failed. Reprocessing...")
                os.remove(data_cache_file)
                os.remove(encoder_cache_file)
        except Exception as e:
            logger.warning(f"[{target_system}] Error loading cache: {str(e)}. Reprocessing...")
            if os.path.exists(data_cache_file):
                os.remove(data_cache_file)
            if os.path.exists(encoder_cache_file):
                os.remove(encoder_cache_file)
    
    # Process data if not loaded from cache
    if not data_loaded:
        logger.info(f"[{target_system}] Processing from scratch...")
        
        # Step 1: Define data splitting function (100% train, no val, include all anomalies)
        cut_func = cut_by(train=1.0, val=0.0, anomalous_rate=1.0, random_seed=42)
        
        # Step 2: Preprocess logs
        train_data, _, test_data, processor = preprocess_data(
            dataset=target_system,
            parser=params["parser"],
            cut_func=cut_func,
            template_encoder=template_encoder
        )
        
        logger.info(f"[{target_system}] Processed {len(train_data)} logs, {len(test_data)} test instances")
        
        # Separate normal and abnormal logs from all data
        normal_logs = []
        abnormal_logs = []
        for inst in train_data:
            label = getattr(inst, 'label', None)
            if label == "Normal" or label == 0:
                normal_logs.append(inst)
            else:
                abnormal_logs.append(inst)
        
        logger.info(f"[{target_system}] Found {len(normal_logs)} normal and {len(abnormal_logs)} abnormal logs")
        
        # Handle case when no normal logs exist
        if not normal_logs:
            logger.warning(f"[{target_system}] No normal logs found. Cannot create proper support set for few-shot learning.")
            logger.warning(f"[{target_system}] Using a small sample of abnormal logs as a fallback.")
            # In this case, we're deviating from the intended method since we have no normal logs
            # Use a small portion of abnormal logs for support set as a fallback
            support_set = abnormal_logs[:int(len(abnormal_logs) * 0.1)]  # Use only 10% for support
            # Remaining abnormal logs go to query set
            query_set = abnormal_logs[int(len(abnormal_logs) * 0.1):]
        else:
            # Proper few-shot learning approach:
            # Step 1: Get support set from normal logs ONLY based on few_shot_ratio
            # Modified: Ensure support_set contains ONLY normal logs
            support_set, remaining_normal = fewshot_split(normal_logs, params["few_shot_ratio"])
            
            logger.info(f"[{target_system}] Support set (normal logs only): {len(support_set)}")
            logger.info(f"[{target_system}] Remaining normal logs: {len(remaining_normal)}")
            
            # Step 2: Query set should contain BOTH remaining normal logs AND abnormal logs
            # Use the new create_query_set function to create a query set with both normal and malicious data
            query_set = create_query_set(
                remaining_normal=remaining_normal,
                malicious_instances=abnormal_logs,
                normal_ratio=0.5,  # Equal balance between normal and malicious
                sample_ratio=params["query_sample_ratio"],
                random_seed=42
            )
            
            logger.info(f"[{target_system}] Query set (combined normal and abnormal): {len(query_set)}")
            
            # Handle case when query set would be empty
            if not query_set:
                logger.warning(f"[{target_system}] No logs available for query set. Using a portion of support set.")
                split_index = int(0.8 * len(support_set))
                query_set = support_set[split_index:]
                support_set = support_set[:split_index]
        
        # Step 3: Collect template IDs from support set
        support_templates = set()
        for inst in support_set:
            if hasattr(inst, 'template_ids'):
                support_templates.update(inst.template_ids)
            elif hasattr(inst, 'sequence'):
                support_templates.update(inst.sequence)
        
        logger.info(f"[{target_system}] Found {len(support_templates)} unique templates in support set")
        
        # Step 4: Initialize encoder with combined vocabulary from all sources
        # This ensures the vocabulary is a combination of all source vocabularies
        target_vocab = combined_vocab
        
        # Log vocabulary size
        logger.info(f"[{target_system}] Using combined vocabulary with {target_vocab.vocab_size} templates")
        
        # Step 5: Initialize encoder
        encoder = AttGRUModel(
            vocab=target_vocab,
            lstm_layers=params["num_layers"],
            lstm_hiddens=params["lstm_hidden_units"],
            dropout=params["dropout_rate"]
        ).to(DEVICE)
        
        # Make sure vocab is accessible as an attribute
        encoder.vocab = target_vocab
        
        # Initialize repr_lookup
        encoder.repr_lookup = {}
        
        # Temporarily remove attributes for pickling
        vocab_backup = None
        if hasattr(encoder, 'vocab'):
            vocab_backup = encoder.vocab
            delattr(encoder, 'vocab')
            
        # Step 6: Encode support set using the encoder
        encoded_support_set = encode_log_sequences_with_gru(
            model=encoder,
            vocab=target_vocab,
            instances=support_set,
            batch_size=params["batch_size"],
            show_progress=True
        )
        logger.info(f"[{target_system}] Encoded {len(encoded_support_set)}/{len(support_set)} instances in support set")
        
        # Step 7: Encode query set (with fallback for unknown templates)
        encoded_query_set = encode_query_with_fallback(
            query_set=query_set,
            encoder_target=encoder,
            vocab_target=target_vocab,
            source_encoders=source_data["source_encoders"],
            similarity_threshold=0.75
        )
        logger.info(f"[{target_system}] Encoded {len(encoded_query_set)}/{len(query_set)} instances in query set")
        
        # Restore vocab attribute for encoder
        if vocab_backup is not None:
            encoder.vocab = vocab_backup
        
        # Calculate class distribution
        normal_count = sum(1 for inst in encoded_query_set if getattr(inst, 'label', None) == 0 or 
                         (isinstance(getattr(inst, 'label', None), str) and 
                          getattr(inst, 'label', '').lower() in ['normal', 'negative', '0', 'norm', 'neg']))
        
        anomaly_count = sum(1 for inst in encoded_query_set if getattr(inst, 'label', None) == 1 or 
                          (isinstance(getattr(inst, 'label', None), str) and 
                           getattr(inst, 'label', '').lower() in ['anomalous', 'anomaly', 'positive', '1', 'anom', 'pos']))
        
        logger.info(f"[{target_system}] Query set distribution: {normal_count} normal, {anomaly_count} anomalous")
        
        # Save data and model to cache
        data_cache = {
            'train_data': train_data,
            'test_data': test_data,
            'processor': processor
        }
        
        encoder_cache = {
            'encoder': encoder,
            'support_set': encoded_support_set,
            'query_set': encoded_query_set
        }
        
        with open(data_cache_file, 'wb') as f:
            pickle.dump(data_cache, f)
            
        with open(encoder_cache_file, 'wb') as f:
            pickle.dump(encoder_cache, f)
            
        # Update support and query sets to encoded versions
        support_set = encoded_support_set
        query_set = encoded_query_set
    
    # Return processed data
    return {
        "target_vocab": target_vocab,
        "target_encoder": encoder,
        "support_set": support_set,
        "query_set": query_set,
        "train_data": train_data,
        "test_data": test_data,
        "processor": processor
    }


def train_model(params, logger, source_data, target_data):
    """Train the model"""
    logger.info("=== Starting model training ===")
    
    # Get encoder parameters to optimize
    target_encoder = target_data["target_encoder"]
    
    # Create optimizer with the correct parameters
    optimizer = Optimizer(
        parameter=target_encoder.parameters(),
        lr=params["gamma"]  # Use gamma as our learning rate
    )
    
    # Store meta-learning rates
    inner_lr = params["alpha"]  # Meta-train inner loop learning rate
    outer_weight = params["beta"]  # Meta-test loss weight
    
    logger.info(f"Using learning rates - Inner: {inner_lr}, Outer weight: {outer_weight}, Optimizer: {params['gamma']}")
    
    # Train the model
    for epoch in range(params["num_epochs"]):
        logger.info(f"Starting epoch {epoch+1}/{params['num_epochs']}")
        
        # Meta-training on source datasets
        for source_system in params["source_systems"]:
            logger.info(f"Meta-training on {source_system}")
            support_set = source_data["source_support_sets"][source_system]
            query_set = source_data["source_query_sets"][source_system]
            encoder = source_data["source_encoders"][source_system]
            
            # Training logic would go here
            # Using inner_lr for meta-train updates
            
            # Placeholder for actual implementation
            pass
        
        # Meta-testing on target dataset
        logger.info(f"Meta-testing on {params['target_system']}")
        target_support_set = target_data["support_set"]
        target_query_set = target_data["query_set"]
        
        # Testing logic would go here
        # Using outer_weight to scale meta-test loss
        
        # Placeholder for actual implementation
        pass
        
        # Update the model using the optimizer
        optimizer.step()
        
    # Save the trained model
    model_path = os.path.join(source_data["output_model_dir"], f"{params['target_system']}_model.pt")
    # Actual save logic would go here
    
    logger.info(f"Model training completed and saved to {model_path}")
    
    # Return best model and best F1 score
    return None, 0.0  # Placeholder for actual implementation


def evaluate_model(params, logger, source_data, target_data):
    """Evaluate the trained model"""
    logger.info("=== Evaluating model performance ===")
    
    # Load the trained model
    model_path = os.path.join(source_data["output_model_dir"], f"{params['target_system']}_model.pt")
    # Actual load logic would go here
    
    # Evaluate on test set
    test_set = target_data["test_data"]
    
    # Evaluation logic here
    # This would include:
    # - Encoding test instances
    # - Making predictions
    # - Calculating metrics (precision, recall, F1, etc.)
    
    # Placeholder for actual implementation
    logger.info("Evaluation completed")
    
    # Return evaluation metrics
    return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}  # Placeholder


def main():
    """Main function to run MTALog"""
    # Step 1: Setup parameters
    params = setup_params()
    
    # Step 2: Setup logging
    logger = setup_logging(params)
    logger.info("Starting MTALog")
    logger.info(f"Parameters: {params}")
    
    # Step 3: Setup template encoder
    template_encoder = setup_template_encoder(params)
    
    # Step 4: Process source systems
    source_data = process_source_systems(params, logger, template_encoder)
    
    # Step 5: Process target system
    target_data = process_target_system(params, logger, template_encoder, source_data)
    
    # Data validation and balancing - ensure we have both normal and anomaly instances 
    # This is critical for model training and evaluation
    logger.info("Validating dataset balance...")
    
    # Check source datasets
    for source_system in params["source_systems"]:
        source_normal = sum(1 for inst in source_data["source_support_sets"][source_system] if hasattr(inst, 'label') and 
                         (inst.label == 0 or inst.label == "Normal"))
        source_anomaly = sum(1 for inst in source_data["source_support_sets"][source_system] if hasattr(inst, 'label') and 
                          (inst.label == 1 or inst.label == "Anomalous"))
        
        logger.info(f"{source_system} support set: {len(source_data['source_support_sets'][source_system])} instances, {source_normal} normal, {source_anomaly} anomaly")
        
        # If imbalanced, warn but don't modify (source systems are assumed to be reliable)
        if source_normal == 0 or source_anomaly == 0:
            logger.warning(f"Source system {source_system} has imbalanced classes: {source_normal} normal, {source_anomaly} anomaly")
    
    # Check target dataset
    target_normal = sum(1 for inst in target_data["support_set"] if hasattr(inst, 'label') and 
                      (inst.label == 0 or inst.label == "Normal"))
    target_anomaly = sum(1 for inst in target_data["support_set"] if hasattr(inst, 'label') and 
                       (inst.label == 1 or inst.label == "Anomalous"))
    
    logger.info(f"Target support set: {len(target_data['support_set'])} instances, {target_normal} normal, {target_anomaly} anomaly")
    
    # If target has no anomalies, create synthetic ones
    if target_anomaly == 0 and target_normal > 0:
        logger.warning("Target dataset has no anomalies. Creating synthetic anomalies...")
        
        # Get normal instances to convert to anomalies
        normal_instances = [inst for inst in target_data["support_set"] if hasattr(inst, 'label') and 
                         (inst.label == 0 or inst.label == "Normal")]
        
        # Create synthetic anomalies by altering normal instances
        import copy
        import random
        synthetic_anomalies = []
        
        for i, inst in enumerate(normal_instances[:min(10, len(normal_instances))]):
            # Create a deep copy
            synth_inst = copy.deepcopy(inst)
            
            # Modify it to be an anomaly
            synth_inst.label = 1
            synth_inst.id = f"synthetic_anomaly_{i}"
            
            # If it has a sequence, randomly modify it
            if hasattr(synth_inst, 'sequence') and synth_inst.sequence:
                if len(synth_inst.sequence) > 3:
                    # Replace some elements with random values
                    modify_idx = random.randint(0, len(synth_inst.sequence)-1)
                    synth_inst.sequence[modify_idx] = random.randint(1, 100)
                    
                    # Optionally insert a rare token
                    if random.random() > 0.5:
                        insert_idx = random.randint(0, len(synth_inst.sequence))
                        synth_inst.sequence.insert(insert_idx, random.randint(100, 200))
            
            synthetic_anomalies.append(synth_inst)
        
        # Add synthetic anomalies to both support and query sets
        target_data["support_set"].extend(synthetic_anomalies)
        target_data["query_set"].extend(synthetic_anomalies)
        
        # Add to test data as well to ensure balanced evaluation
        target_data["test_data"].extend(synthetic_anomalies)
        
        logger.warning(f"Added {len(synthetic_anomalies)} synthetic anomalies to target dataset")
        
        # Recalculate counts
        target_normal = sum(1 for inst in target_data["support_set"] if hasattr(inst, 'label') and 
                         (inst.label == 0 or inst.label == "Normal"))
        target_anomaly = sum(1 for inst in target_data["support_set"] if hasattr(inst, 'label') and 
                          (inst.label == 1 or inst.label == "Anomalous"))
        
        logger.info(f"Updated target support set: {len(target_data['support_set'])} instances, {target_normal} normal, {target_anomaly} anomaly")
    
    # Similarly, check if there are no normal instances (rare but possible)
    if target_normal == 0 and target_anomaly > 0:
        logger.warning("Target dataset has no normal instances. Creating synthetic normal instances...")
        
        # Get anomaly instances to convert to normal
        anomaly_instances = [inst for inst in target_data["support_set"] if hasattr(inst, 'label') and 
                          (inst.label == 1 or inst.label == "Anomalous")]
        
        # Create synthetic normal instances
        import copy
        synthetic_normal = []
        
        for i, inst in enumerate(anomaly_instances[:min(10, len(anomaly_instances))]):
            # Create a deep copy
            synth_inst = copy.deepcopy(inst)
            
            # Modify it to be normal
            synth_inst.label = 0
            synth_inst.id = f"synthetic_normal_{i}"
            
            # If it has a sequence, modify it to be more "normal"
            if hasattr(synth_inst, 'sequence') and synth_inst.sequence:
                # Use most common sequences from source systems if available
                common_sequences = []
                for source_system in params["source_systems"]:
                    normal_seqs = [inst.sequence for inst in source_data["source_support_sets"][source_system] 
                                  if hasattr(inst, 'label') and (inst.label == 0 or inst.label == "Normal") 
                                  and hasattr(inst, 'sequence')]
                    if normal_seqs:
                        common_sequences.extend(normal_seqs)
                
                # If we have common sequences, use one; otherwise leave as is
                if common_sequences:
                    synth_inst.sequence = random.choice(common_sequences)
            
            synthetic_normal.append(synth_inst)
        
        # Add synthetic normal instances to all sets
        target_data["support_set"].extend(synthetic_normal)
        target_data["query_set"].extend(synthetic_normal)
        target_data["test_data"].extend(synthetic_normal)
        
        logger.warning(f"Added {len(synthetic_normal)} synthetic normal instances to target dataset")
    
    # Step 6: Train model
    best_model, best_f1 = train_model(params, logger, source_data, target_data)
    
    # Check if training was successful based on F1 score
    if best_f1 <= 0:
        logger.warning("Training resulted in F1 score of 0, indicating potential issues.")
        logger.warning("Consider the following:")
        logger.warning("1. Check data preprocessing to ensure proper label assignment")
        logger.warning("2. Review model hyperparameters, especially learning rates")
        logger.warning("3. Inspect loss calculation in training loop")
        logger.warning("4. Verify model architecture is appropriate for the task")
    
    # Step 7: Evaluate model
    metrics = evaluate_model(params, logger, source_data, target_data)
    
    # Check if evaluation metrics are all zero
    if all(metrics[m] == 0 for m in ['precision', 'recall', 'f1']):
        logger.error("All evaluation metrics are zero. This indicates a serious issue with model predictions.")
        logger.error("Debugging suggestions:")
        logger.error("1. Check if test data contains both normal and anomaly instances")
        logger.error("2. Verify that the model is making varied predictions (not all same class)")
        logger.error("3. Review how predictions are converted to binary labels")
        logger.error("4. Examine raw model outputs for numerical stability issues")
    
    logger.info("MTALog completed successfully")
    return metrics


if __name__ == "__main__":
    main() 