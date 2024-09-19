import re
import sys

import matplotlib.pyplot as plt

STATISTICS_TEMPLATE_LOG_PATH = "logs/Statistics_Template.log"
METALOG_LOG_PATH = "logs/MetaLog.log"

session = sys.argv[1]
word2vec_file = "glove.840B.300d.txt"

with open(STATISTICS_TEMPLATE_LOG_PATH, "r") as file:
    lines = file.readlines()

    for line in lines:
        regex = rf"^.+ - Statistics_Template_Encoder - {session} - INFO: Loading word2vec dict from (.+)\.$"
        match = re.search(regex, line)

        if match is not None:
            word2vec_file = match.group().split()[-1]
            break

TITLE = f"BILATERAL GENERALIZATION TRANSFERRING HDFS TO BGL\n(using {word2vec_file})\n"

meta_train_losses = []
meta_test_losses = []
with open(METALOG_LOG_PATH, "r") as file:
    lines = file.readlines()

    for line in lines:
        regex = rf"^.* - MetaLog - {session} - INFO: Step: (.+) \| Epoch: (.+) \| Meta-train loss: (.+) \| Meta-test loss: (.+)\.$"
        match = re.search(regex, line)

        if match is not None:
            params = match.group().split("|")

            meta_train_loss = float(params[2].split()[-1].split()[-1])
            meta_train_losses.append(meta_train_loss)

            meta_test_loss = float(params[3].split()[-1].split()[-1][:-1])
            meta_test_losses.append(meta_test_loss)


fig, axs = plt.subplots(1, 1, figsize=(16, 8))
num_validations = [i * 500 for i in range(len(meta_train_losses))]

axs.set_ylim(0, 1)
axs.plot(num_validations, meta_train_losses, color="tab:blue")
axs.plot(num_validations, meta_test_losses, color="tab:orange")
axs.legend(["Meta-train loss", "Meta-test loss"])
axs.set_xlabel("Step")
axs.set_ylabel("Loss")

fig.suptitle(f"{TITLE}\n" + f"\n")

fig.savefig(f"visualization/loss/{word2vec_file}-{session}.png")
