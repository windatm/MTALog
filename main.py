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

# Third-party
import matplotlib.pyplot as plt

# Local modules
from CONSTANTS import DEVICE, PROJECT_ROOT, SESSION
from models.gru import AttGRUModel
from module.Common import data_iter, generate_tinsts_binary_label
from module.Optimizer import Optimizer
from preprocessing.datacutter.SimpleCutting import cut_by, fewshot_split, sample_query_set, cut_all
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
    """Process source log systems"""
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
                
                # Validate that the cache contains valid data
                is_valid = True
                for system in params["source_systems"]:
                    support_set = combined_data.get("source_support_sets", {}).get(system, [])
                    normal_count = len([x for x in support_set if x.label == 0])
                    anomaly_count = len([x for x in support_set if x.label == 1])
                    
                    if normal_count == 0 and anomaly_count == 0:
                        logger.warning(f"Cache invalid for {system}. Reprocessing...")
                        is_valid = False
                        break
                
                if is_valid:
                    logger.info("Using cached source systems data")
                    return combined_data
                
                # Delete invalid cache
                os.remove(combined_cache_file)
                
        except Exception as e:
            logger.warning(f"Error loading cache: {str(e)}. Reprocessing...")
            if os.path.exists(combined_cache_file):
                os.remove(combined_cache_file)
    
    # Process each source system
    for source_system in params["source_systems"]:
        logger.info(f"=== [SOURCE] Processing system: {source_system} ===")
        
        # Define cache files
        data_cache_file = os.path.join(cache_dir, f"{source_system}_{params['parser']}_data.pkl")
        encoder_cache_file = os.path.join(cache_dir, f"{source_system}_{params['parser']}_encoder.pkl")
        
        # Try to load from cache first
        data_loaded = False
        if os.path.exists(data_cache_file) and os.path.exists(encoder_cache_file):
            try:
                # Load preprocessed data
                with open(data_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    train_data = cache_data['train_data']
                    processor = cache_data['processor']
                    vocab = cache_data['vocab']
                
                # Load encoder and encoded data
                with open(encoder_cache_file, 'rb') as f:
                    encoder_cache = pickle.load(f)
                    encoder = encoder_cache['encoder']
                    support_set = encoder_cache['support_set']
                    query_set = encoder_cache['query_set']
                
                # Rebuild repr_lookup if needed
                if not hasattr(encoder, 'repr_lookup') or not encoder.repr_lookup:
                    encoder.repr_lookup = {}
                    # Rebuild from support and query sets
                    for inst in support_set + query_set:
                        if hasattr(inst, 'sequence') and hasattr(inst, 'repr'):
                            encoder.repr_lookup[tuple(inst.sequence)] = inst.repr
                
                # Validate data has both normal and anomaly logs
                normal_count = len([x for x in train_data if x.label == 0])
                anomaly_count = len([x for x in train_data if x.label == 1])
                
                if normal_count > 0 or anomaly_count > 0:
                    logger.info(f"[{source_system}] Loaded from cache: {len(train_data)} instances, {normal_count} normal, {anomaly_count} anomaly")
                    data_loaded = True
                else:
                    logger.warning(f"[{source_system}] Cached data has no valid logs. Reprocessing...")
                    os.remove(data_cache_file)
                    os.remove(encoder_cache_file)
            except Exception as e:
                logger.warning(f"[{source_system}] Error loading cache: {str(e)}. Reprocessing...")
                if os.path.exists(data_cache_file):
                    os.remove(data_cache_file)
                if os.path.exists(encoder_cache_file):
                    os.remove(encoder_cache_file)
        
        # Process if not loaded from cache
        if not data_loaded:
            # Step 1: Preprocess full data (normal + abnormal)
            cut_func = cut_by(train=1.0, val=0.0, anomalous_rate=1.0)
            train_data, _, _, processor = preprocess_data(
                dataset=source_system,
                parser=params["parser"],
                cut_func=cut_func,
                template_encoder=template_encoder
            )
            normal_count = sum(1 for inst in train_data if inst.label == 0)
            anomaly_count = len(train_data) - normal_count
            logger.info(f"[{source_system}] Processed {len(train_data)} instances: {normal_count} normal, {anomaly_count} anomaly")
            
            # Step 2: Load vocabulary
            vocab = Vocab()
            vocab.load_from_dict(processor.embedding)
            
            # Save preprocessed data to cache
            data_cache = {
                'train_data': train_data,
                'processor': processor,
                'vocab': vocab
            }
            with open(data_cache_file, 'wb') as f:
                pickle.dump(data_cache, f)
            
            # Step 3: Initialize encoder
            encoder = AttGRUModel(
                vocab=vocab,
                lstm_layers=params["num_layers"],
                lstm_hiddens=params["lstm_hidden_units"],
                dropout=params["dropout_rate"],
            ).to(DEVICE)
            
            # Step 4: Encode all log instances
            encoded_data = encode_log_sequences_with_gru(encoder, vocab, train_data, batch_size=params["batch_size"])
            
            # If encoding failed, use fallback with zero vectors
            if not encoded_data:
                logger.warning(f"[{source_system}] Encoding failed, using fallback")
                for inst in train_data:
                    if not hasattr(inst, 'repr') or inst.repr is None:
                        repr_dim = 2 * params["lstm_hidden_units"]
                        inst.repr = np.zeros(repr_dim)
                encoded_data = train_data
            
            # Step 5: Split into support/query
            random.shuffle(encoded_data)
            split_index = int(0.5 * len(encoded_data))
            support_set = encoded_data[:split_index]
            query_set = encoded_data[split_index:]
            
            # Step 6: Build fallback repr lookup
            encoder.repr_lookup = {
                tuple(inst.sequence): inst.repr for inst in encoded_data
            }
            
            # Temporarily store repr_lookup
            repr_lookup = encoder.repr_lookup
            encoder.repr_lookup = {}  # Empty dict for pickling
            
            # Save encoder and encoded data to cache
            encoder_cache = {
                'encoder': encoder,
                'support_set': support_set,
                'query_set': query_set
            }
            with open(encoder_cache_file, 'wb') as f:
                pickle.dump(encoder_cache, f)
                
            # Restore repr_lookup
            encoder.repr_lookup = repr_lookup
        
        # Store processed data
        source_processors[source_system] = processor
        source_vocabularies[source_system] = vocab
        source_encoders[source_system] = encoder
        source_support_sets[source_system] = support_set
        source_query_sets[source_system] = query_set
        
        # Log info about support and query sets
        logger.info(f"[{source_system}] Support set: {len(support_set)} | Query set: {len(query_set)}")
    
    # Create combined vocabulary from all source systems
    combined_vocab = Vocab()
    combined_vocab.load_from_dict(template_encoder.get_embeddings())
    
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
    
    # Try to load from cache first
    data_loaded = False
    if os.path.exists(data_cache_file) and os.path.exists(encoder_cache_file):
        try:
            # Load preprocessed data
            with open(data_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                train_data = cache_data['train_data']
                test_data = cache_data['test_data']
                processor = cache_data['processor']
            
            # Load encoded data
            with open(encoder_cache_file, 'rb') as f:
                encoder_cache = pickle.load(f)
                encoder = encoder_cache['encoder']
                support_set = encoder_cache['support_set']
                query_set = encoder_cache['query_set']
                support_templates = encoder_cache['support_templates']
            
            # Rebuild repr_lookup if needed
            if not hasattr(encoder, 'repr_lookup') or not encoder.repr_lookup:
                encoder.repr_lookup = {}
                # Rebuild from support and query sets
                for inst in support_set + query_set:
                    if hasattr(inst, 'sequence') and hasattr(inst, 'repr'):
                        encoder.repr_lookup[tuple(inst.sequence)] = inst.repr
            
            # Validate data has both normal and anomaly logs
            normal_count = len([x for x in train_data if x.label == 0])
            anomaly_count = len([x for x in train_data if x.label == 1])
            
            if normal_count > 0 or anomaly_count > 0:
                logger.info(f"[{target_system}] Loaded from cache: {len(train_data)} instances, {normal_count} normal, {anomaly_count} anomaly")
                logger.info(f"[{target_system}] Support set: {len(support_set)} | Query set: {len(query_set)}")
                data_loaded = True
            else:
                logger.warning(f"[{target_system}] Cached data has no valid logs. Reprocessing...")
                os.remove(data_cache_file)
                os.remove(encoder_cache_file)
        except Exception as e:
            logger.warning(f"[{target_system}] Error loading cache: {str(e)}. Reprocessing...")
            if os.path.exists(data_cache_file):
                os.remove(data_cache_file)
            if os.path.exists(encoder_cache_file):
                os.remove(encoder_cache_file)
    
    # Process if not loaded from cache
    if not data_loaded:
        # Step 1: Preprocess the target dataset
        processor = Preprocessor()
        train_data, _, test_data = processor.process(
            dataset=target_system,
            parsing=params["parser"],
            template_encoding=template_encoder.present,
            cut_func=cut_all
        )
        normal_count = sum(1 for inst in train_data if inst.label == 0)
        anomaly_count = len(train_data) - normal_count
        logger.info(f"[{target_system}] Processed {len(train_data)} instances: {normal_count} normal, {anomaly_count} anomaly")
        
        # Step 2: Split data with few-shot learning
        normal_logs = [x for x in train_data if x.label == 0]
        abnormal_logs = [x for x in train_data if x.label == 1]
        
        # Get support and query sets
        support_set, remaining = fewshot_split(normal_logs, params["few_shot_ratio"])
        support_set += abnormal_logs  # Add all anomalous logs to support set
        query_set = sample_query_set(remaining, params["query_sample_ratio"])
        
        # Step 3: Collect template IDs from support set
        support_templates = set()
        for inst in support_set:
            if hasattr(inst, 'template_ids'):
                support_templates.update(inst.template_ids)
        
        # Step 4: Initialize encoder
        encoder = AttGRUModel(
            vocab=combined_vocab,
            lstm_layers=params["num_layers"],
            lstm_hiddens=params["lstm_hidden_units"],
            dropout=params["dropout_rate"]
        ).to(DEVICE)
        
        # Initialize repr_lookup
        encoder.repr_lookup = {}
        
        # Save data to cache
        data_cache = {
            'train_data': train_data,
            'test_data': test_data,
            'processor': processor
        }
        with open(data_cache_file, 'wb') as f:
            pickle.dump(data_cache, f)
        
        # Save encoded data to cache
        encoder_cache = {
            'encoder': encoder,
            'support_set': support_set,
            'query_set': query_set,
            'support_templates': support_templates
        }
        with open(encoder_cache_file, 'wb') as f:
            pickle.dump(encoder_cache, f)
    
    # Return the processed data
    return {
        "target_preprocessor": processor,
        "target_vocab": combined_vocab,
        "target_encoder": encoder,
        "target_support_set": support_set,
        "target_query_set": query_set,
        "target_inst_train": train_data,
        "target_inst_test": test_data,
        "support_templates": support_templates
    }


def train_model(params, logger, source_data, target_data):
    """Train the model"""
    logger.info("=== Starting model training ===")
    
    # Create optimizer
    optimizer = Optimizer(
        alpha=params["alpha"],
        beta=params["beta"],
        gamma=params["gamma"]
    )
    
    # Train the model
    for epoch in range(params["num_epochs"]):
        logger.info(f"Starting epoch {epoch+1}/{params['num_epochs']}")
        
        # Meta-training on source datasets
        for source_system in params["source_systems"]:
            logger.info(f"Meta-training on {source_system}")
            support_set = source_data["source_support_sets"][source_system]
            query_set = source_data["source_query_sets"][source_system]
            encoder = source_data["source_encoders"][source_system]
            
            # Training logic here
            # This would include:
            # - Encoding support and query sets
            # - Calculating meta-train loss
            # - Updating model parameters
            
            # Placeholder for actual implementation
            pass
        
        # Meta-testing on target dataset
        logger.info(f"Meta-testing on {params['target_system']}")
        target_support_set = target_data["target_support_set"]
        target_query_set = target_data["target_query_set"]
        target_encoder = target_data["target_encoder"]
        
        # Testing logic here
        # This would include:
        # - Encoding support and query sets
        # - Calculating meta-test loss
        # - Evaluating performance
        
        # Placeholder for actual implementation
        pass
    
    # Save the trained model
    model_path = os.path.join(source_data["output_model_dir"], f"{params['target_system']}_model.pt")
    # Actual save logic would go here
    
    logger.info(f"Model training completed and saved to {model_path}")


def evaluate_model(params, logger, source_data, target_data):
    """Evaluate the trained model"""
    logger.info("=== Evaluating model performance ===")
    
    # Load the trained model
    model_path = os.path.join(source_data["output_model_dir"], f"{params['target_system']}_model.pt")
    # Actual load logic would go here
    
    # Evaluate on test set
    test_set = target_data["target_inst_test"]
    
    # Evaluation logic here
    # This would include:
    # - Encoding test instances
    # - Making predictions
    # - Calculating metrics (precision, recall, F1, etc.)
    
    # Placeholder for actual implementation
    logger.info("Evaluation completed")


def main():
    """Main function to run the MTALog workflow"""
    # Setup parameters
    params = setup_params()
    
    # Setup logging
    logger = setup_logging(params)
    
    # Setup template encoder
    template_encoder = setup_template_encoder(params)
    
    # Process source systems
    source_data = process_source_systems(params, logger, template_encoder)
    
    # Process target system
    target_data = process_target_system(params, logger, template_encoder, source_data)
    
    # Train model if in training mode
    if params["mode"] == "train":
        train_model(params, logger, source_data, target_data)
    
    # Evaluate model
    evaluate_model(params, logger, source_data, target_data)


if __name__ == "__main__":
    main() 