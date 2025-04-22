import numpy as np
import logging

logger = logging.getLogger(__name__)

class ResultStatistics:
    """
    Class for storing and calculating evaluation metrics for anomaly detection results.
    
    This class calculates standard evaluation metrics like precision, recall, F1, accuracy,
    and also stores confusion matrix elements (TP, FP, TN, FN).
    
    Attributes:
        true_positive (int): Number of true positives
        false_positive (int): Number of false positives
        true_negative (int): Number of true negatives
        false_negative (int): Number of false negatives
        precision (float): Precision score
        recall (float): Recall score
        f1 (float): F1 score
        accuracy (float): Accuracy score
    """
    
    def __init__(self):
        """Initialize statistics with zero values."""
        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        self.accuracy = 0.0
    
    def calculate(self, y_true, y_pred):
        """
        Calculate statistics based on true labels and predictions.
        
        Args:
            y_true (list): Ground truth labels (0 for normal, 1 for anomaly)
            y_pred (list): Predicted labels (0 for normal, 1 for anomaly)
            
        Returns:
            self: The updated statistics object
        """
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: {len(y_true)} vs {len(y_pred)}")
            
        self.true_positive = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        self.false_positive = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        self.true_negative = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        self.false_negative = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        
        # Calculate precision
        if self.true_positive + self.false_positive > 0:
            self.precision = self.true_positive / (self.true_positive + self.false_positive)
        else:
            self.precision = 0.0
            
        # Calculate recall
        if self.true_positive + self.false_negative > 0:
            self.recall = self.true_positive / (self.true_positive + self.false_negative)
        else:
            self.recall = 0.0
            
        # Calculate F1 score
        if self.precision + self.recall > 0:
            self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        else:
            self.f1 = 0.0
            
        # Calculate accuracy
        total = len(y_true)
        if total > 0:
            self.accuracy = (self.true_positive + self.true_negative) / total
        else:
            self.accuracy = 0.0
            
        return self
    
    def update(self, other_stats):
        """
        Update this statistics object with values from another.
        
        Args:
            other_stats (ResultStatistics): Another statistics object
            
        Returns:
            self: The updated statistics object
        """
        self.true_positive += other_stats.true_positive
        self.false_positive += other_stats.false_positive
        self.true_negative += other_stats.true_negative
        self.false_negative += other_stats.false_negative
        
        # Recalculate metrics
        if self.true_positive + self.false_positive > 0:
            self.precision = self.true_positive / (self.true_positive + self.false_positive)
        else:
            self.precision = 0.0
            
        if self.true_positive + self.false_negative > 0:
            self.recall = self.true_positive / (self.true_positive + self.false_negative)
        else:
            self.recall = 0.0
            
        if self.precision + self.recall > 0:
            self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        else:
            self.f1 = 0.0
            
        total = self.true_positive + self.false_positive + self.true_negative + self.false_negative
        if total > 0:
            self.accuracy = (self.true_positive + self.true_negative) / total
        else:
            self.accuracy = 0.0
            
        return self
    
    def to_dict(self):
        """
        Convert statistics to a dictionary.
        
        Returns:
            dict: Dictionary with statistic values
        """
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tp": self.true_positive,
            "fp": self.false_positive,
            "fn": self.false_negative,
            "tn": self.true_negative
        }
    
    def __str__(self):
        """Return string representation of the statistics."""
        return (f"Accuracy: {self.accuracy:.4f}, Precision: {self.precision:.4f}, "
                f"Recall: {self.recall:.4f}, F1: {self.f1:.4f}, "
                f"TP: {self.true_positive}, FP: {self.false_positive}, "
                f"TN: {self.true_negative}, FN: {self.false_negative}") 