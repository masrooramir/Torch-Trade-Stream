import os
import sys
import torch
import random
import numpy as np


# DIRS
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
print(f"Settings initialized. Project's base path setted - {BASE_DIR}")

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models_and_weights")
PREDICTIONS = os.path.join(BASE_DIR, "data", "model_predictions")


# # MODEL PARMS
# SEQUENCE_LENGTH = 60
# BATCH_SIZE = 32
# LEARNING_RATE = 0.001
# EPOCHS = 50
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# INIT  
def init_workspace():
    target_dirs = [RAW_DIR, PROCESSED_DIR, MODELS_DIR, PREDICTIONS]

    for folder in target_dirs:
        os.makedirs(folder, exist_ok=True)
        gitkeep_file = os.path.join(folder, ".gitkeep")
        if not os.path.exists(gitkeep_file):
            with open(gitkeep_file, 'w') as f:
                pass

# PERSISTS MODEL BEHAVIOUR
# def enforce_determinism(seed=42):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False