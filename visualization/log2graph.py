import argparse
import re
import sys

import matplotlib.pyplot as plt

PATH = "logs/MetaLog.log"
session = sys.argv[1]

argparser = argparse.ArgumentParser()

argparser.add_argument(
    "--word2vec_file", default="glove.6B.300d.txt", type=str, help="GloVe file"
)

args, extra_args = argparser.parse_known_args()

word2vec_file = args.word2vec_file
TITLE = f"Bilateral generalization experiments transferring using {word2vec_file}"

f1_scores = []
train_losses = []
test_losses = []
with open(PATH, "r") as file:
    lines = file.readlines()
    for line in lines:
        regex = rf"^.* - MetaLog - {session} - INFO: Test BGL: F1 score = (.*) | Precision = (.*) | Recall = (.*)$"
        match = re.search(regex, line)

        if match is not None:
            params = match.group().split(", ")
            f1_score = float(params[1].split()[1])

            f1_scores.append(f1_score)

    for line in lines:
        regex = rf"^.* - MetaLog - {session} - INFO: Step:.*, Epoch:.*, meta train loss:.*, meta test loss:.*$"
        match = re.search(regex, line)

        if match is not None:
            params = match.group().split(", ")
            train_loss = float(params[2].split()[-1].split(":")[-1])
            train_losses.append(train_loss)

            test_loss = float(params[3].split()[-1].split(":")[-1])
            test_losses.append(test_loss)


fig, axs = plt.subplots(2, 1, figsize=(16, 8))
REMOVE_LAST_TWO = 2  # remove the last two data points for Best model and Last model
num_epoches = [i for i in range(len(f1_scores) - REMOVE_LAST_TWO)]
num_validations = [i * 500 for i in range(len(train_losses))]

axs[0].set_ylim(0, 100)
axs[0].plot(num_epoches, f1_scores[:-REMOVE_LAST_TWO])
axs[0].legend(["F1 Score"])
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Test")

max_f1_score = max(f1_scores)
for i in range(len(num_epoches)):
    axs[0].text(
        num_epoches[i],
        f1_scores[i] + 5,
        round(f1_scores[i], 2),
        color="red" if f1_scores[i] == max_f1_score else "black",
        ha="center",
    )

axs[1].set_ylim(0, 1)
axs[1].plot(num_validations, train_losses, num_validations, test_losses)
axs[1].legend(["Meta-train loss", "Meta-test loss"])
axs[1].set_xlabel("Step")
axs[1].set_ylabel("Loss")

LOSS_EPS = 0.05
min_test_loss = min(test_losses)
for i in range(0, len(num_validations), 2):
    axs[1].text(
        num_validations[i], train_losses[i] + LOSS_EPS, train_losses[i], ha="center"
    )
    axs[1].text(
        num_validations[i],
        test_losses[i] + LOSS_EPS,
        test_losses[i],
        color="red" if test_losses[i] == min_test_loss else "black",
        ha="center",
    )

LAST_MODEL_INDEX = -2
BEST_MODEL_INDEX = -1
fig.suptitle(
    f"{TITLE}\n"
    + f"Last model: F1 Score = {f1_scores[LAST_MODEL_INDEX]}\n"
    + f"Best model: F1 Score = {f1_scores[BEST_MODEL_INDEX]}"
)

fig.savefig(f"visualization/graphs/{word2vec_file}-{session}.png")
