import argparse
import re
import sys

import matplotlib.pyplot as plt

PATH = "logs/MetaLog.log"
SESSION = sys.argv[1]

argparser = argparse.ArgumentParser()

argparser.add_argument("--source", default="HDFS", type=str, help="HDFS or BGL")
argparser.add_argument("--target", default="BGL", type=str, help="HDFS or BGL")
argparser.add_argument(
    "--word2vec", default="glove.6B.300d.txt", type=str, help="GloVe file"
)

args, extra_args = argparser.parse_known_args()

SOURCE = args.source
TARGET = args.target
WORD2VEC_FILE = args.word2vec
TITLE = f"Bilateral generalization experiments transferring \nfrom {SOURCE} to {TARGET} using {WORD2VEC_FILE}"

f1_scores = []
train_losses = []
test_losses = []
with open(PATH, "r") as file:
    lines = file.readlines()
    for line in lines:
        regex = rf"^.* - MetaLog - {SESSION} - INFO: Precision = .* \/ .* = (.*), Recall = .* \/ .* = (.*) F1 score = (.*), FPR = (.*)$"
        match = re.search(regex, line)

        if match is not None:
            params = match.group().split(", ")
            f1_score = float(params[1].split()[-1])

            f1_scores.append(f1_score)

    for line in lines:
        regex = rf"^.* - MetaLog - {SESSION} - INFO: Step:.*, Epoch:.*, meta train loss:.*, meta test loss:.*$"
        match = re.search(regex, line)

        if match is not None:
            params = match.group().split(", ")
            train_loss = float(params[2].split()[-1].split(":")[-1])
            train_losses.append(train_loss)

            test_loss = float(params[3].split()[-1].split(":")[-1])
            test_losses.append(test_loss)


fig, axs = plt.subplots(2, 1, figsize=(16, 8))
num_epoches = [i for i in range(len(f1_scores))]
num_validations = [i * 500 for i in range(len(train_losses))]

axs[0].set_ylim(0, 100)
axs[0].plot(num_epoches, f1_scores)
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

fig.suptitle(TITLE)

fig.savefig(f"graphs/{SESSION}.png")
