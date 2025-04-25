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
import json
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from main import (
    setup_params,
    setup_logging,
    setup_template_encoder,
    process_source_systems,
    process_target_system,
    analyze_datasets
)
from training import train_model, evaluate_model, predict
from CONSTANTS import DEVICE, PROJECT_ROOT
from module.Optimizer import Optimizer
from utils.analysis import (
    analyze_templates, analyze_vocabulary, analyze_overlap,
    analyze_representations, create_log_analysis_plots, export_dataset_stats
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='MTALog - Meta-Transfer Learning for Log Anomaly Detection')
    
    # Mode Selection
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'predict', 'analysis'],
                       help='Operation mode: train, evaluate, predict, or analysis')
    
    # System Configuration
    parser.add_argument('--source_systems', type=str, nargs='+', default=['HDFS', 'OpenStack'],
                       help='Source log systems for meta-training')
    parser.add_argument('--target_system', type=str, default='BGL',
                       help='Target log system for meta-testing')
    parser.add_argument('--parser', type=str, default='IBM', choices=['IBM', 'Drain', 'Spell'],
                       help='Log parser to use')
    
    # Training Parameters
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--few_shot_ratio', type=float, default=0.1,
                       help='Ratio of logs used for few-shot learning')
    parser.add_argument('--query_sample', type=float, default=1.0,
                       help='Ratio of query set to use for evaluation')
    
    # Analysis Parameters
    parser.add_argument('--analysis_output', type=str, default='analysis_results',
                       help='Directory to save analysis results')

    return parser.parse_args()

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def analysis_mode(params, logger):
    """Run analysis mode to stop at preprocessing and analyze data"""
    logger.info("=== Running in ANALYSIS mode ===")
    
    # Setup template encoder for processing log data
    template_encoder = setup_template_encoder(params)
    
    # Process source systems
    logger.info("Processing source systems for analysis...")
    source_data = process_source_systems(params, logger, template_encoder)
    
    # Process target system
    logger.info("Processing target system for analysis...")
    target_data = process_target_system(params, logger, template_encoder, source_data)
    
    # Create output directory for analysis results
    analysis_dir = os.path.join(PROJECT_ROOT, params.get('analysis_output', 'analysis_results'))
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Perform analysis
    logger.info("Starting data analysis...")
    
    # Analyze templates in source and target systems
    template_analysis_results = {}
    for system, support_set in source_data.get("source_support_sets", {}).items():
        template_analysis_results[system] = analyze_templates(support_set, logger)
    
    target_template_analysis = analyze_templates(target_data.get("target_support_set", {}), logger)
    template_analysis_results[params["target_system"]] = target_template_analysis
    
    # Save template analysis results
    with open(os.path.join(analysis_dir, "template_analysis.json"), "w") as f:
        json.dump(template_analysis_results, f, indent=2)
    
    # Analyze vocabulary
    vocab_analysis_results = {}
    for system, vocab in source_data.get("source_vocabularies", {}).items():
        vocab_analysis_results[system] = analyze_vocabulary(vocab, logger)
    
    combined_vocab_analysis = analyze_vocabulary(source_data.get("combined_vocab", {}), logger)
    vocab_analysis_results["combined"] = combined_vocab_analysis
    
    # Save vocabulary analysis results
    with open(os.path.join(analysis_dir, "vocabulary_analysis.json"), "w") as f:
        json.dump(vocab_analysis_results, f, indent=2)
    
    # Analyze vocabulary overlap between source systems and target
    overlap_results = {}
    for system, vocab in source_data.get("source_vocabularies", {}).items():
        if target_data and "target_vocab" in target_data:
            overlap = analyze_overlap(vocab, target_data["target_vocab"], logger)
            overlap_results[f"{system}_to_{params['target_system']}"] = overlap
    
    # Save vocabulary overlap results
    with open(os.path.join(analysis_dir, "vocabulary_overlap.json"), "w") as f:
        json.dump(overlap_results, f, indent=2)
    
    # Create visualizations
    logger.info("Creating visualization plots...")
    plot_dir = os.path.join(analysis_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    created_plots = create_log_analysis_plots(source_data, target_data, plot_dir)
    
    # Export dataset statistics to CSV
    logger.info("Exporting dataset statistics...")
    csv_dir = os.path.join(analysis_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    csv_files = export_dataset_stats(source_data, target_data, csv_dir)
    
    # Generate comprehensive report
    logger.info("Generating analysis report...")
    create_analysis_report(template_analysis_results, vocab_analysis_results, 
                          overlap_results, created_plots, csv_files, analysis_dir)
    
    logger.info(f"Analysis complete. Results saved to {analysis_dir}")
    return source_data, target_data

def create_analysis_report(template_analysis, vocab_analysis, overlap_results, 
                          plots, csv_files, output_dir):
    """Create an HTML report summarizing analysis results"""
    report_path = os.path.join(output_dir, "analysis_report.html")
    
    # Create simple HTML report
    with open(report_path, "w") as f:
        f.write("<html>\n<head>\n")
        f.write("<title>MTALog Data Analysis Report</title>\n")
        f.write("<style>\n")
        f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
        f.write("h1, h2, h3 { color: #2c3e50; }\n")
        f.write("table { border-collapse: collapse; width: 100%; }\n")
        f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
        f.write("th { background-color: #f2f2f2; }\n")
        f.write("img { max-width: 800px; margin: 10px 0; }\n")
        f.write(".section { margin-bottom: 30px; }\n")
        f.write("</style>\n</head>\n<body>\n")
        
        # Report header
        f.write("<h1>MTALog Data Analysis Report</h1>\n")
        f.write(f"<p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
        
        # Template Analysis Section
        f.write("<div class='section'>\n")
        f.write("<h2>Template Analysis</h2>\n")
        f.write("<table>\n")
        f.write("<tr><th>System</th><th>Total Templates</th><th>Unique Templates</th>")
        f.write("<th>Avg Templates Per Log</th><th>Anomaly-Only Templates</th></tr>\n")
        
        for system, results in template_analysis.items():
            f.write(f"<tr><td>{system}</td>")
            f.write(f"<td>{results.get('total_templates', 'N/A')}</td>")
            f.write(f"<td>{results.get('unique_templates', 'N/A')}</td>")
            
            # Handle case where templates_per_log_avg might be a string or doesn't exist
            avg_templates = results.get('templates_per_log_avg', 'N/A')
            if isinstance(avg_templates, (int, float)):
                f.write(f"<td>{avg_templates:.2f}</td>")
            else:
                f.write(f"<td>{avg_templates}</td>")
            
            f.write(f"<td>{len(results.get('templates_only_in_anomalous', []))}</td></tr>\n")
        
        f.write("</table>\n")
        f.write("</div>\n")
        
        # Vocabulary Analysis Section
        f.write("<div class='section'>\n")
        f.write("<h2>Vocabulary Analysis</h2>\n")
        f.write("<table>\n")
        f.write("<tr><th>System</th><th>Vocabulary Size</th><th>Embedding Dimension</th></tr>\n")
        
        for system, results in vocab_analysis.items():
            f.write(f"<tr><td>{system}</td>")
            f.write(f"<td>{results.get('vocab_size', 'N/A')}</td>")
            f.write(f"<td>{results.get('embedding_dim', 'N/A')}</td></tr>\n")
        
        f.write("</table>\n")
        f.write("</div>\n")
        
        # Vocabulary Overlap Section
        f.write("<div class='section'>\n")
        f.write("<h2>Vocabulary Overlap Analysis</h2>\n")
        f.write("<table>\n")
        f.write("<tr><th>Systems</th><th>Overlap Size</th><th>Overlap Percentage</th></tr>\n")
        
        for systems, results in overlap_results.items():
            f.write(f"<tr><td>{systems}</td>")
            f.write(f"<td>{results.get('overlap_size', 'N/A')}</td>")
            f.write(f"<td>{results.get('overlap_percentage', 'N/A'):.2f}%</td></tr>\n")
        
        f.write("</table>\n")
        f.write("</div>\n")
        
        # Plots Section
        if plots:
            f.write("<div class='section'>\n")
            f.write("<h2>Visualizations</h2>\n")
            
            for plot_path in plots:
                plot_name = os.path.basename(plot_path)
                rel_path = os.path.join("plots", plot_name)
                f.write(f"<h3>{plot_name.replace('.png', '').replace('_', ' ').title()}</h3>\n")
                f.write(f"<img src='{rel_path}' alt='{plot_name}'>\n")
            
            f.write("</div>\n")
        
        # CSV Files Section
        if csv_files:
            f.write("<div class='section'>\n")
            f.write("<h2>Dataset Statistics</h2>\n")
            f.write("<p>The following CSV files were generated:</p>\n")
            f.write("<ul>\n")
            
            for csv_path in csv_files:
                csv_name = os.path.basename(csv_path)
                f.write(f"<li><a href='csv/{csv_name}'>{csv_name}</a></li>\n")
            
            f.write("</ul>\n")
            f.write("</div>\n")
        
        f.write("</body>\n</html>")

def main():
    """Main function"""
    # Set seed for reproducibility
    set_seed(42)
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Configure parameters based on command-line arguments
    params = setup_params()
    
    # Update parameters from command-line arguments
    params['mode'] = args.mode
    params['source_systems'] = args.source_systems
    params['target_system'] = args.target_system
    params['parser'] = args.parser
    params['num_epochs'] = args.epochs
    params['few_shot_ratio'] = args.few_shot_ratio
    params['query_sample_ratio'] = args.query_sample
    params['analysis_output'] = args.analysis_output
    
    # Setup logging
    logger = setup_logging(params)
    
    # Process according to mode
    if args.mode == 'analysis':
        # Run analysis mode - stop at preprocessing and analyze data
        source_data, target_data = analysis_mode(params, logger)
    elif args.mode == 'train':
        # Setup template encoder
        template_encoder = setup_template_encoder(params)
        
        # Process source systems
        source_data = process_source_systems(params, logger, template_encoder)
        
        # Process target system
        target_data = process_target_system(params, logger, template_encoder, source_data)
        
        # Train the model
        train_model(params, logger, source_data, target_data)
    elif args.mode == 'evaluate':
        # Setup template encoder
        template_encoder = setup_template_encoder(params)
        
        # Process source systems
        source_data = process_source_systems(params, logger, template_encoder)
        
        # Process target system
        target_data = process_target_system(params, logger, template_encoder, source_data)
        
        # Evaluate the model
        results = evaluate_model(params, logger, source_data, target_data)
        
        # Print evaluation results
        logger.info(f"Evaluation results: {results}")
    elif args.mode == 'predict':
        # Run in prediction mode
        logger.info("Prediction mode not fully implemented yet.")

if __name__ == '__main__':
    main()