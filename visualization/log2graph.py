import os
import re
import sys

import matplotlib.pyplot as plt

# Constants
STATISTICS_TEMPLATE_LOG_PATH = "logs/Statistics_Template.log"
MTALOG_LOG_PATH = "logs/MTALog.log"
LOSS_EPS = 0.05

# Get session from command line arguments
session = sys.argv[1]


def extract_word2vec_file(log_path, session):
    """Extract the word2vec file path from the statistics template log."""
    pattern = rf"^.+ - Statistics_Template_Encoder - {session} - INFO: Loading word2vec dict from (.+)\.$"
    with open(log_path, "r") as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                return match.group(1)
    return None


def extract_meta_log_data(log_path, session):
    """Extract losses, parameters, and F1 scores from MTALog."""
    meta_train_losses, meta_test_losses = [], []
    train_f1_scores, test_f1_scores = [], []
    params = {"lstm_hiddens": None, "num_layer": None, "drop_out": None, "lr": None}

    patterns = {
        "lstm_hiddens": rf"^.* - MTALog - {session} - INFO:   - LSTM hidden units: (.+)$",
        "num_layer": rf"^.* - MTALog - {session} - INFO:   - Number of layers: (.+)$",
        "drop_out": rf"^.* - MTALog - {session} - INFO:   - Dropout rate: (.+)$",
        "lr": rf"^.* - MTALog - {session} - INFO:   - Learning rate: (.+)$",
        "loss": rf"^.* - MTALog - {session} - INFO: Step: .+ \| Epoch: .+ \| Meta-train loss: (.+) \| Meta-test loss: (.+)\.$",
        "train": rf"^.+ - MTALog - {session} - INFO: Train: F1 score = (.+) \| Precision = .+ \| Recall = .+$",
        "test": rf"^.+ - MTALog - {session} - INFO: Test: F1 score = (.+) \| Precision = .+ \| Recall = .+$",
    }

    with open(log_path, "r") as file:
        for line in file:
            for key in params:
                if params[key] is None and (match := re.search(patterns[key], line)):
                    params[key] = match.group(1)
            if match := re.search(patterns["loss"], line):
                meta_train_losses.append(float(match.group(1)))
                meta_test_losses.append(float(match.group(2)))
            if match := re.search(patterns["train"], line):
                train_f1_scores.append(float(match.group(1)))
            if match := re.search(patterns["test"], line):
                test_f1_scores.append(float(match.group(1)))

    return meta_train_losses, meta_test_losses, train_f1_scores, test_f1_scores, params


def plot_f1_scores(ax, num_epochs, train_f1, test_f1):
    """Plot train and test F1 scores on the provided axis."""
    ax.set_ylim(0, 110)
    ax.plot(num_epochs, train_f1, color="tab:blue", marker="o", label="Train")
    ax.plot(num_epochs, test_f1, color="tab:orange", marker="o", label="Test")
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")

    for i, (train, test) in enumerate(zip(train_f1, test_f1)):
        ax.text(num_epochs[i], train + 5, round(train, 2), ha="center")
        ax.text(num_epochs[i], test - 10, round(test, 2), ha="center")


def plot_meta_losses(ax, num_steps, meta_train_losses, meta_test_losses):
    """Plot meta-train and meta-test losses on the provided axis."""
    ax.plot(num_steps, meta_train_losses, color="tab:blue", label="Meta-train loss")
    ax.plot(num_steps, meta_test_losses, color="tab:orange", label="Meta-test loss")
    ax.legend()
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")


# Main execution
if __name__ == "__main__":
    word2vec_file = extract_word2vec_file(STATISTICS_TEMPLATE_LOG_PATH, session)
    title = f"BILATERAL GENERALIZATION TRANSFERRING HDFS TO BGL USING {word2vec_file}"

    meta_train_losses, meta_test_losses, train_f1_scores, test_f1_scores, params = (
        extract_meta_log_data(MTALOG_LOG_PATH, session)
    )

    fig, axs = plt.subplots(2, 1, figsize=(16, 8))
    num_epochs = list(range(len(train_f1_scores)))
    num_steps = [i * 10 for i in range(len(meta_train_losses))]

    plot_f1_scores(axs[0], num_epochs, train_f1_scores, test_f1_scores)
    plot_meta_losses(axs[1], num_steps, meta_train_losses, meta_test_losses)

    # Set the title for the plot
    best_test_f1_score = max(test_f1_scores)
    fig_title = (
        f"{title}\nBest model F1 Score = {best_test_f1_score}\n"
        f"LSTM hidden units = {params['lstm_hiddens']} | Layers = {params['num_layer']} | "
        f"Drop out = {params['drop_out']} | Learning rate = {params['lr']}"
    )
    fig.suptitle(fig_title)

    # Define the path to save the plot
    plot_dir = os.path.join("visualization", "graphs")
    plot_filename = f"{word2vec_file}-{session}.png"
    plot_path = os.path.join(plot_dir, plot_filename)

    # Ensure the directory exists
    os.makedirs(plot_dir, exist_ok=True)

    # Save the plot
    fig.savefig(plot_path)
