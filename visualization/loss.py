import re
import sys

import matplotlib.pyplot as plt

# Constants
STATISTICS_TEMPLATE_LOG_PATH = "logs/Statistics_Template.log"
METALOG_LOG_PATH = "logs/MetaLog.log"

# Get session from command line arguments
session = sys.argv[1]

# Default word2vec file
word2vec_file = "glove.840B.300d.txt"


# Function to extract word2vec file from statistics log
def extract_word2vec_file(log_path, session):
    with open(log_path, "r") as file:
        for line in file:
            regex = rf"^.+ - Statistics_Template_Encoder - {session} - INFO: Loading word2vec dict from (.+)\.$"
            match = re.search(regex, line)
            if match:
                return match.group(1)
    return word2vec_file


# Extract word2vec file
word2vec_file = extract_word2vec_file(STATISTICS_TEMPLATE_LOG_PATH, session)

# Title for the plot
TITLE = f"BILATERAL GENERALIZATION TRANSFERRING HDFS TO BGL\n(using {word2vec_file})\n"


# Function to extract losses and parameters from MetaLog
def extract_losses_and_params(log_path, session):
    meta_train_losses = []
    meta_test_losses = []
    alpha = beta = gamma = None

    with open(log_path, "r") as file:
        for line in file:
            if re.search(rf"^.* - MetaLog - {session} - INFO:   - Alpha: .*$", line):
                alpha = line.split()[-1]
            if re.search(rf"^.* - MetaLog - {session} - INFO:   - Beta: .*$", line):
                beta = line.split()[-1]
            if re.search(rf"^.* - MetaLog - {session} - INFO:   - Gamma: .*$", line):
                gamma = line.split()[-1]

            loss_match = re.search(
                rf"^.* - MetaLog - {session} - INFO: Step: (.+) \| Epoch: (.+) \| Meta-train loss: (.+) \| Meta-test loss: (.+)\.$",
                line,
            )
            if loss_match:
                params = loss_match.group().split("|")
                meta_train_loss = float(params[2].split()[-1])
                meta_test_loss = float(params[3].split()[-1][:-1])
                meta_train_losses.append(meta_train_loss)
                meta_test_losses.append(meta_test_loss)

    return meta_train_losses, meta_test_losses, alpha, beta, gamma


# Extract losses and parameters
meta_train_losses, meta_test_losses, alpha, beta, gamma = extract_losses_and_params(
    METALOG_LOG_PATH, session
)

# Plotting
fig, axs = plt.subplots(1, 1, figsize=(16, 8))
num_validations = [i * 500 for i in range(len(meta_train_losses))]

axs.set_ylim(0, 1)
axs.plot(num_validations, meta_train_losses, color="tab:blue")
axs.plot(num_validations, meta_test_losses, color="tab:orange")
axs.legend(["Meta-train loss", "Meta-test loss"])
axs.set_xlabel("Step")
axs.set_ylabel("Loss")

fig.suptitle(f"{TITLE}\nAlpha: {alpha} | Beta: {beta} | Gamma: {gamma}\n")

# Save the plot
fig.savefig(f"visualization/loss/{word2vec_file}-{session}.png")
