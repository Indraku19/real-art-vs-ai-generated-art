"""Central project configuration, paths, label conventions, and global seed locking."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# Project directory layout
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OOD_DIR = DATA_DIR / "ood"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENT_LOG_PATH = EXPERIMENTS_DIR / "experiment_log.csv"

# Reproducibility & dataset constants
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
HF_DATASET_ID = "DataScienceProject/Art_Images_Ai_And_Real_"

# Explicit binary label mapping
LABEL_MAP = {0: "Real Art", 1: "AI-Generated Art"}
CLASS_TO_IDX = {"Real Art": 0, "AI-Generated Art": 1}


def set_global_seed(seed: int = SEED) -> None:
    """Lock Python, NumPy, and TensorFlow random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass
