#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command-line interface for MTALog system
"""

import os
import sys
import argparse
import torch
import random

from main import (
    setup_params,
    setup_logging,
    setup_template_encoder,
    process_source_systems,
    process_target_system
)
from training import train_model, evaluate_model, predict
from CONSTANTS import DEVICE, PROJECT_ROOT
from module.Optimizer import Optimizer
from models.gru import AttGRUModel
from preprocessing.Preprocess import Preprocessor
from utils.common import get_model_and_result_paths
from utils.data_processing import create_embedding_lookup
from preprocessing.datacutter.SimpleCutting import cut_all, cut_by, fewshot_split, sample_query_set
from utils.vocab import Vocab


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="MTALog: Meta-Transfer Learning for Log Anomaly Detection")
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'predict'],
                        help='Operation mode: train, eval, or predict')
    
    # System selection
    parser.add_argument('--source_systems', type=str, nargs='+', default=['HDFS', 'OpenStack'],
                        help='Source log systems to use for meta-learning')
    parser.add_argument('--target_system', type=str, default='BGL',
                        help='Target log system for anomaly detection')
    
    # Parser selection
    parser.add_argument('--parser', type=str, default='IBM', choices=['IBM', 'Drain', 'Spell'],
                        help='Log parser to use')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Hidden size of the GRU encoder')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of GRU layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--alpha', type=float, default=8e-3,
                        help='Inner loop learning rate (meta-train)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Outer loop scaling factor (meta-test loss weight)')
    parser.add_argument('--gamma', type=float, default=8e-3,
                        help='Learning rate for optimizer')
    
    # Few-shot learning parameters
    parser.add_argument('--few_shot_ratio', type=float, default=0.1,
                        help='Ratio of normal logs used in support (e.g., 0.1 for 10%)')
    parser.add_argument('--query_sample_ratio', type=float, default=1.0,
                        help='Ratio of query set sampled for evaluation')
    
    # Model checkpoint
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved model checkpoint (for eval and predict modes)')
    
    # Data parameters
    parser.add_argument('--word2vec_file', type=str, default='glove.6B.300d.txt',
                        help='Word2Vec embeddings file')
    
    # Parse the arguments
    args = parser.parse_args()
    return args


def update_params_from_args(params, args):
    """Update parameter dictionary with command-line arguments"""
    # Update model parameters
    params['lstm_hidden_units'] = args.hidden_size
    params['num_layers'] = args.num_layers
    params['dropout_rate'] = args.dropout
    
    # Update training parameters
    params['batch_size'] = args.batch_size
    params['num_epochs'] = args.epochs
    params['alpha'] = args.alpha
    params['beta'] = args.beta
    params['gamma'] = args.gamma
    
    # Update system parameters
    params['source_systems'] = args.source_systems
    params['target_system'] = args.target_system
    params['parser'] = args.parser
    
    # Update few-shot parameters
    params['few_shot_ratio'] = args.few_shot_ratio
    params['query_sample_ratio'] = args.query_sample_ratio
    
    # Update mode
    params['mode'] = args.mode
    
    # Update word2vec file
    params['word2vec_file'] = args.word2vec_file
    
    return params


def train_mode(args):
    """Execute training mode"""
    # Setup parameters
    params = setup_params()
    params = update_params_from_args(params, args)
    
    # Setup logging
    logger = setup_logging(params)
    logger.info(f"Running in train mode with target system: {params['target_system']}")
    
    # Setup template encoder
    template_encoder = setup_template_encoder(params)
    
    # Process source systems
    source_data = process_source_systems(params, logger, template_encoder)
    
    # Process target system
    target_data = process_target_system(params, logger, template_encoder, source_data)
    
    # Create optimizer with correct parameters
    if hasattr(target_data["target_encoder"], "parameters"):
        optimizer = Optimizer(
            parameter=target_data["target_encoder"].parameters(),
            lr=params['gamma']
        )
    else:
        logger.error("Error: target_encoder does not have a parameters method")
        return
    
    # Train model
    best_model, best_f1 = train_model(
        source_systems=params['source_systems'],
        source_support_sets=source_data['source_support_sets'],
        source_query_sets=source_data['source_query_sets'],
        target_support_set=target_data['target_support_set'],
        target_query_set=target_data['target_query_set'],
        source_encoders=source_data['source_encoders'],
        target_encoder=target_data['target_encoder'],
        optimizer=optimizer,
        device=DEVICE,
        num_epochs=params['num_epochs'],
        batch_size=params['batch_size'],
        output_model_dir=source_data['output_model_dir'],
        logger=logger
    )
    
    # Final evaluation
    final_metrics = evaluate_model(
        test_data=target_data['target_inst_test'],
        encoder=target_data['target_encoder'],
        optimizer=optimizer,
        device=DEVICE,
        batch_size=params['batch_size'],
        logger=logger
    )
    
    logger.info("Training complete!")
    return final_metrics


def eval_mode(args):
    """Execute evaluation mode"""
    if args.model_path is None:
        print("Error: model_path must be provided in eval mode")
        sys.exit(1)
    
    # Setup parameters
    params = setup_params()
    params = update_params_from_args(params, args)
    
    # Setup logging
    logger = setup_logging(params)
    logger.info(f"Running in eval mode with target system: {params['target_system']}")
    
    # Setup template encoder
    template_encoder = setup_template_encoder(params)
    
    # Process target system only
    _, output_res_dir = get_model_and_result_paths(params["parser"], PROJECT_ROOT)
    
    # Create target preprocessor
    target_preprocessor = Preprocessor()
    target_inst_train, _, target_inst_test = target_preprocessor.process(
        dataset=params['target_system'], 
        parsing=params["parser"], 
        template_encoding=template_encoder.present,
        cut_func=cut_all
    )
    
    # Create a vocabulary for the target system
    target_vocab = Vocab()
    target_vocab.load_from_dict(template_encoder.get_embeddings())
    
    # Create target encoder
    target_encoder = AttGRUModel(
        vocab=target_vocab,
        lstm_layers=params["num_layers"],
        lstm_hiddens=params["lstm_hidden_units"],
        dropout=params["dropout_rate"]
    )
    
    # Load model weights
    if os.path.exists(args.model_path):
        logger.info(f"Loading model from {args.model_path}")
        target_encoder.load_state_dict(torch.load(args.model_path))
    else:
        logger.error(f"Model file {args.model_path} not found")
        sys.exit(1)
    
    # Create optimizer
    optimizer = Optimizer(
        parameter=target_encoder.parameters(),
        lr=params['gamma']
    )
    
    # Evaluate model
    metrics = evaluate_model(
        test_data=target_inst_test,
        encoder=target_encoder,
        optimizer=optimizer,
        device=DEVICE,
        batch_size=params['batch_size'],
        logger=logger
    )
    
    logger.info("Evaluation complete!")
    return metrics


def predict_mode(args):
    """Execute prediction mode"""
    if args.model_path is None:
        print("Error: model_path must be provided in predict mode")
        sys.exit(1)
    
    # Setup parameters
    params = setup_params()
    params = update_params_from_args(params, args)
    
    # Setup logging
    logger = setup_logging(params)
    logger.info(f"Running in predict mode with target system: {params['target_system']}")
    
    # Setup template encoder
    template_encoder = setup_template_encoder(params)
    
    # Create target preprocessor
    target_preprocessor = Preprocessor()
    
    # Process a small set of logs from the target system
    _, _, example_logs = target_preprocessor.process(
        dataset=params['target_system'], 
        parsing=params["parser"], 
        template_encoding=template_encoder.present,
        cut_func=cut_all
    )
    
    # Limit to a small sample for demonstration
    if len(example_logs) > 10:
        example_logs = random.sample(example_logs, 10)
    
    # Create a vocabulary for the target system
    target_vocab = Vocab()
    target_vocab.load_from_dict(template_encoder.get_embeddings())
    
    # Load model
    target_encoder = AttGRUModel(
        vocab=target_vocab,
        lstm_layers=params["num_layers"],
        lstm_hiddens=params["lstm_hidden_units"],
        dropout=params["dropout_rate"]
    )
    
    # Load model weights
    if os.path.exists(args.model_path):
        logger.info(f"Loading model from {args.model_path}")
        target_encoder.load_state_dict(torch.load(args.model_path))
    else:
        logger.error(f"Model file {args.model_path} not found")
        sys.exit(1)
    
    logger.info(f"Loaded {len(example_logs)} log sequences for prediction")
    
    # Create template lookup
    template_lookup = create_embedding_lookup(
        templates=[t for log in example_logs for t in log.template_ids],
        encoder=target_encoder,
        device=DEVICE
    )
    
    # Make predictions
    logger.info(f"Making predictions on {len(example_logs)} log sequences")
    predictions, scores = predict(
        log_sequences=example_logs,
        encoder=target_encoder,
        template_lookup=template_lookup,
        device=DEVICE
    )
    
    # Output results
    logger.info("Prediction Results:")
    for i, (pred, score) in enumerate(zip(predictions, scores)):
        status = "ANOMALY" if pred == 1 else "NORMAL"
        logger.info(f"Log {i+1}: {status} (Score: {score:.4f})")
    
    logger.info("Prediction complete!")
    return predictions, scores


def main():
    """Main function to run the CLI"""
    # Parse command-line arguments
    args = parse_args()
    
    # Execute the appropriate mode
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'eval':
        eval_mode(args)
    elif args.mode == 'predict':
        predict_mode(args)
    else:
        print(f"Invalid mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main() 