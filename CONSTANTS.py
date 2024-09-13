import hashlib
import os
import random
import time

import numpy as np
import torch

seed = 6
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
elif hasattr(torch.mps, "is_available") and torch.mps.is_available():
    torch.mps.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps:0")

SESSION = hashlib.md5(
    time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() + 8 * 60 * 60)).encode(
        "utf-8"
    )
).hexdigest()
SESSION = "SESSION_" + SESSION


def get_project_root():
    # goto the root folder of LogBar
    current_abspath = os.path.abspath("__file__")
    while True:
        if os.path.split(current_abspath)[1] == "MetaLog":
            project_root = current_abspath
            break
        else:
            current_abspath = os.path.dirname(current_abspath)
    return project_root


def get_logs_root():
    log_file_root = os.path.join(get_project_root(), "logs")
    if not os.path.exists(log_file_root):
        os.makedirs(log_file_root)
    return log_file_root


LOG_ROOT = get_logs_root()
PROJECT_ROOT = get_project_root()

pretrained_mode_path = os.path.join(PROJECT_ROOT, "outputs/models/pretrain")
if not os.path.exists(pretrained_mode_path):
    os.makedirs(pretrained_mode_path)
os.environ["TRANSFORMERS_CACHE"] = pretrained_mode_path
os.environ["TRANSFORMERS_OFFLINE"] = "1"
