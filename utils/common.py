import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support


def metrics(y_pred, y_true):
    """Calucate evaluation metrics for precision, recall, and f1.

    Arguments
    ---------
        y_pred: ndarry, the predicted result list
        y_true: ndarray, the ground truth label list

    Returns
    -------
        precision: float, precision value
        recall: float, recall value
        f1: float, f1 measure value
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    return precision, recall, f1


def not_empty(s):
    return s and s.strip()

def get_model_and_result_paths(parser: str, project_root: str) -> tuple[str, str]:
    """
    Generate absolute paths for:
        - Trained model checkpoint,
        - Prediction results.

    Args:
        parser (str): Parser name (e.g., "IBM").
        project_root (str): Root directory of the project.

    Returns:
        tuple[str, str]: 
            - output_model_dir: Directory for trained model checkpoints.
            - output_res_dir: Directory for model prediction results.
    """
    output_model_dir = os.path.join(project_root, "outputs", "models", "MTALog", parser, "model")
    output_res_dir = os.path.join(project_root, "outputs", "results", "MTALog", parser, "detect_res")
    
    return output_model_dir, output_res_dir