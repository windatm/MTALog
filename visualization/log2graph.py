import re
import sys

import matplotlib.pyplot as plt

STATISTICS_TEMPLATE_LOG_PATH = "logs/Statistics_Template.log"
METALOG_LOG_PATH = "logs/MetaLog.log"

session = sys.argv[1]
word2vec_file = "glove.840d.300d.txt"

with open(STATISTICS_TEMPLATE_LOG_PATH, "r") as file:
    lines = file.readlines()

    for line in lines:
        regex = rf"^.+ - Statistics_Template_Encoder - {session} - INFO: Loading word2vec dict from (.+)\.$"
        match = re.search(regex, line)

        if match is not None:
            word2vec_file = match.group().split()[-1]
            break

TITLE = f"BILATERAL GENERALIZATION TRANSFERRING HDFS TO BGL\nusing {word2vec_file})\n"

train_f1_scores = []
test_f1_scores = []
meta_train_losses = []
meta_test_losses = []
with open(METALOG_LOG_PATH, "r") as file:
    lines = file.readlines()

    for line in lines:
        regex = rf"^.+ - MetaLog - {session} - INFO: Train: F1 score = (.+) \| Precision = (.+) \| Recall = (.+)$"
        match = re.search(regex, line)

        if match is not None:
            params = match.group().split("|")

            f1_score = float(params[0].split()[-1])
            train_f1_scores.append(f1_score)

    for line in lines:
        regex = rf"^.+ - MetaLog - {session} - INFO: Test: F1 score = (.+) \| Precision = (.+) \| Recall = (.+)$"
        match = re.search(regex, line)

        if match is not None:
            params = match.group().split("|")

            f1_score = float(params[0].split()[-1])
            test_f1_scores.append(f1_score)

    for line in lines:
        regex = rf"^.* - MetaLog - {session} - INFO: Step: (.+) \| Epoch: (.+) \| Meta-train loss: (.+) \| Meta-test loss: (.+)\.$"
        match = re.search(regex, line)

        if match is not None:
            params = match.group().split("|")

            meta_train_loss = float(params[2].split()[-1].split()[-1])
            meta_train_losses.append(meta_train_loss)

            meta_test_loss = float(params[3].split()[-1].split()[-1][:-1])
            meta_test_losses.append(meta_test_loss)


fig, axs = plt.subplots(2, 1, figsize=(16, 8))
num_epoches = [i for i in range(len(train_f1_scores))]
num_validations = [i * 500 for i in range(len(meta_train_losses))]

axs[0].set_ylim(0, 110)
axs[0].plot(num_epoches, train_f1_scores, color="tab:blue")
axs[0].plot(num_epoches, test_f1_scores, color="tab:orange")
axs[0].legend(["Train", "Test"])
axs[0].set_xlabel("F1 Score")
axs[0].set_ylabel("Test")

max_test_f1_score = max(test_f1_scores)
best_test_f1_score = 0

for i in range(len(num_epoches)):
    if test_f1_scores[i] == max_test_f1_score:
        best_test_f1_score = i
    axs[0].plot(num_epoches[i], train_f1_scores[i], "o", color="tab:blue", zorder=10)
    axs[0].text(
        num_epoches[i],
        train_f1_scores[i] + 5,
        round(train_f1_scores[i], 2),
        color="red" if train_f1_scores[i] == train_f1_scores else "black",
        ha="center",
    )

    axs[0].plot(num_epoches[i], test_f1_scores[i], "o", color="tab:orange", zorder=10)
    axs[0].text(
        num_epoches[i],
        test_f1_scores[i] - 10,
        round(test_f1_scores[i], 2),
        ha="center",
    )

axs[1].set_ylim(0, 1)
axs[1].plot(num_validations, meta_train_losses, color="tab:blue")
axs[1].plot(num_validations, meta_test_losses, color="tab:orange")
axs[1].legend(["Meta-train loss", "Meta-test loss"])
axs[1].set_xlabel("Step")
axs[1].set_ylabel("Loss")

LOSS_EPS = 0.05
min_test_loss = min(meta_test_losses)
for i in range(0, len(num_validations), 2):
    axs[1].plot(
        num_validations[i], meta_train_losses[i], "o", color="tab:blue", zorder=10
    )
    axs[1].text(
        num_validations[i],
        meta_train_losses[i] + LOSS_EPS,
        round(meta_train_losses[i], 4),
        ha="center",
    )

    axs[1].plot(
        num_validations[i], meta_test_losses[i], "o", color="tab:orange", zorder=10
    )
    axs[1].text(
        num_validations[i],
        meta_test_losses[i] + LOSS_EPS,
        round(meta_test_losses[i], 4),
        color="red" if meta_test_losses[i] == min_test_loss else "black",
        ha="center",
    )

fig.suptitle(
    f"{TITLE}\n"
    + f"Last model: F1 Score = {test_f1_scores[-1]}\n"
    + f"Best model: F1 Score = {test_f1_scores[best_test_f1_score]}"
)

fig.savefig(f"visualization/graphs/{word2vec_file}-{session}.png")
