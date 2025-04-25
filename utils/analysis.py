#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for analyzing log data after preprocessing stage
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Dict, List, Any, Set, Tuple

from CONSTANTS import PROJECT_ROOT


def analyze_templates(dataset, logger):
    """
    Analyze templates in a dataset
    
    Args:
        dataset: Dictionary containing log data
        logger: Logger object
        
    Returns:
        Dictionary containing template analysis results
    """
    if not dataset or not isinstance(dataset, dict):
        logger.warning("Invalid dataset provided for template analysis")
        return {}
    
    logs = dataset.get('logs', [])
    labels = dataset.get('labels', [])
    
    if not logs:
        logger.warning("No logs found in the dataset")
        return {}
    
    # Extract templates from logs
    all_templates = []
    templates_per_log = []
    
    for log in logs:
        log_templates = extract_templates_from_log(log)
        all_templates.extend(log_templates)
        templates_per_log.append(len(log_templates))
    
    template_counter = Counter(all_templates)
    
    # Find templates only in anomalous logs
    templates_only_in_anomalous = []
    if len(labels) == len(logs):
        normal_templates = set()
        anomalous_templates = set()
        
        for i, (log, label) in enumerate(zip(logs, labels)):
            log_templates = extract_templates_from_log(log)
            if label == 0:  # Normal
                normal_templates.update(log_templates)
            else:  # Anomalous
                anomalous_templates.update(log_templates)
                
        templates_only_in_anomalous = list(anomalous_templates - normal_templates)
    
    return {
        'total_templates': len(all_templates),
        'unique_templates': len(template_counter),
        'templates_per_log_min': min(templates_per_log) if templates_per_log else 0,
        'templates_per_log_max': max(templates_per_log) if templates_per_log else 0,
        'templates_per_log_avg': sum(templates_per_log) / len(templates_per_log) if templates_per_log else 0,
        'most_common_templates': template_counter.most_common(10),
        'templates_only_in_anomalous': templates_only_in_anomalous
    }


def extract_templates_from_log(log):
    """
    Extract templates from a log entry
    
    Args:
        log: Log entry (can be text or structured data)
        
    Returns:
        List of templates
    """
    if isinstance(log, str):
        # Simple template extraction for string logs
        return [log]
    elif isinstance(log, list):
        # For token-based logs
        return [' '.join(log) if isinstance(log, list) else str(log)]
    elif isinstance(log, dict) and 'template' in log:
        # For structured logs with templates
        return [log['template']]
    elif isinstance(log, dict) and 'content' in log:
        # For structured logs with content
        return [log['content']]
    else:
        # Default case
        return [str(log)]


def analyze_vocabulary(vocabulary, logger):
    """
    Analyze vocabulary statistics
    
    Args:
        vocabulary: Vocabulary object or dictionary
        logger: Logger object
        
    Returns:
        Dictionary containing vocabulary analysis results
    """
    if not vocabulary:
        logger.warning("Invalid vocabulary provided for analysis")
        return {}
    
    vocab_size = 0
    embedding_dim = 0
    token_stats = {}
    
    # Extract vocabulary size
    if hasattr(vocabulary, 'vocab_size'):
        vocab_size = vocabulary.vocab_size
    elif isinstance(vocabulary, dict) and 'vocab_size' in vocabulary:
        vocab_size = vocabulary['vocab_size']
    elif isinstance(vocabulary, dict):
        vocab_size = len(vocabulary)
    
    # Extract embedding dimension if available
    if hasattr(vocabulary, 'embedding_dim'):
        embedding_dim = vocabulary.embedding_dim
    elif isinstance(vocabulary, dict) and 'embedding_dim' in vocabulary:
        embedding_dim = vocabulary['embedding_dim']
    elif isinstance(vocabulary, dict) and 'embeddings' in vocabulary:
        if hasattr(vocabulary['embeddings'], 'shape'):
            embedding_dim = vocabulary['embeddings'].shape[-1]
    
    return {
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'token_stats': token_stats
    }


def analyze_overlap(vocab1, vocab2, logger):
    """
    Analyze overlap between two vocabularies
    
    Args:
        vocab1: First vocabulary
        vocab2: Second vocabulary
        logger: Logger object
        
    Returns:
        Dictionary containing overlap analysis results
    """
    vocab1_tokens = set()
    vocab2_tokens = set()
    
    # Extract tokens from vocab1
    if hasattr(vocab1, 'idx2word'):
        vocab1_tokens = set(vocab1.idx2word.values())
    elif isinstance(vocab1, dict) and 'idx2word' in vocab1:
        vocab1_tokens = set(vocab1['idx2word'].values())
    elif isinstance(vocab1, dict):
        vocab1_tokens = set(vocab1.keys())
    
    # Extract tokens from vocab2
    if hasattr(vocab2, 'idx2word'):
        vocab2_tokens = set(vocab2.idx2word.values())
    elif isinstance(vocab2, dict) and 'idx2word' in vocab2:
        vocab2_tokens = set(vocab2['idx2word'].values())
    elif isinstance(vocab2, dict):
        vocab2_tokens = set(vocab2.keys())
    
    overlap = vocab1_tokens.intersection(vocab2_tokens)
    overlap_percentage = (len(overlap) / len(vocab1_tokens)) * 100 if vocab1_tokens else 0
    
    return {
        'vocab1_size': len(vocab1_tokens),
        'vocab2_size': len(vocab2_tokens),
        'overlap_size': len(overlap),
        'overlap_percentage': overlap_percentage,
        'unique_to_vocab1': len(vocab1_tokens - vocab2_tokens),
        'unique_to_vocab2': len(vocab2_tokens - vocab1_tokens)
    }


def analyze_representations(source_data, target_data, logger):
    """
    Analyze log representations
    
    Args:
        source_data: Dictionary containing source systems data
        target_data: Dictionary containing target system data
        logger: Logger object
        
    Returns:
        Dictionary containing representation analysis results
    """
    results = {}
    
    # Placeholder for representation analysis
    # This can be expanded based on the specific implementation
    
    return results


def create_log_analysis_plots(source_data, target_data, output_dir):
    """
    Create various plots for log analysis
    
    Args:
        source_data: Dictionary containing source systems data
        target_data: Dictionary containing target system data
        output_dir: Directory to save plots
        
    Returns:
        List of created plot paths
    """
    os.makedirs(output_dir, exist_ok=True)
    created_plots = []
    
    # Plot template distribution
    plt.figure(figsize=(12, 8))
    template_counts = []
    system_names = []
    
    for system, support_set in source_data.get("source_support_sets", {}).items():
        if support_set and "logs" in support_set:
            template_counts.append(len(support_set["logs"]))
            system_names.append(system)
    
    if target_data and "target_support_set" in target_data and target_data["target_support_set"] and "logs" in target_data["target_support_set"]:
        template_counts.append(len(target_data["target_support_set"]["logs"]))
        system_names.append(target_data.get("target_system", "Target"))
    
    plt.bar(system_names, template_counts)
    plt.title("Log Template Count by System")
    plt.xlabel("System")
    plt.ylabel("Template Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "template_distribution.png")
    plt.savefig(plot_path)
    plt.close()
    created_plots.append(plot_path)
    
    # More plots can be added here based on specific analysis needs
    
    return created_plots


def export_dataset_stats(source_data, target_data, output_dir):
    """
    Export dataset statistics to CSV files
    
    Args:
        source_data: Dictionary containing source systems data
        target_data: Dictionary containing target system data
        output_dir: Directory to save CSV files
        
    Returns:
        List of created CSV file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    created_files = []
    
    # Export template statistics
    template_stats = []
    for system, support_set in source_data.get("source_support_sets", {}).items():
        if support_set:
            # Handle case where support_set is a list
            if isinstance(support_set, list):
                logs = support_set
                labels = []
                # Try to find labels in source_data if available
                if "source_labels" in source_data and system in source_data["source_labels"]:
                    labels = source_data["source_labels"][system]
                stats = {
                    "system": system,
                    "total_logs": len(logs),
                    "normal_logs": sum(1 for label in labels if label == 0) if labels else "Unknown",
                    "anomalous_logs": sum(1 for label in labels if label == 1) if labels else "Unknown"
                }
            # Handle case where support_set is a dictionary
            else:
                logs = support_set.get("logs", [])
                labels = support_set.get("labels", [])
                stats = {
                    "system": system,
                    "total_logs": len(logs),
                    "normal_logs": sum(1 for label in labels if label == 0),
                    "anomalous_logs": sum(1 for label in labels if label == 1)
                }
            template_stats.append(stats)
    
    if target_data and "target_support_set" in target_data:
        target_support_set = target_data["target_support_set"]
        # Handle case where target_support_set is a list
        if isinstance(target_support_set, list):
            logs = target_support_set
            labels = []
            # Try to find labels in target_data if available
            if "target_labels" in target_data:
                labels = target_data["target_labels"]
            target_stats = {
                "system": target_data.get("target_system", "Target"),
                "total_logs": len(logs),
                "normal_logs": sum(1 for label in labels if label == 0) if labels else "Unknown",
                "anomalous_logs": sum(1 for label in labels if label == 1) if labels else "Unknown"
            }
        # Handle case where target_support_set is a dictionary
        else:
            logs = target_support_set.get("logs", [])
            labels = target_support_set.get("labels", [])
            target_stats = {
                "system": target_data.get("target_system", "Target"),
                "total_logs": len(logs),
                "normal_logs": sum(1 for label in labels if label == 0),
                "anomalous_logs": sum(1 for label in labels if label == 1)
            }
        template_stats.append(target_stats)
    
    if template_stats:
        df = pd.DataFrame(template_stats)
        csv_path = os.path.join(output_dir, "dataset_stats.csv")
        df.to_csv(csv_path, index=False)
        created_files.append(csv_path)
    
    # More CSV exports can be added here based on specific analysis needs
    
    return created_files 