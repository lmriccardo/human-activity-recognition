import torch

from pathlib import Path

# Absolute path to the original and modified dataset
DATASET_PATH           = Path(__file__).absolute().parent.parent.parent
ORIGINAL_DATASET       = DATASET_PATH / "dataset/time_series_data_human_activities.csv"
MODIFIED_DATASET       = DATASET_PATH / "dataset/modified_dataset.csv"
LENGHTS_DATASET        = DATASET_PATH / "dataset/modified_dataset_lenghts.csv"
TRAIN_MODIFIED_DATASET = DATASET_PATH / "dataset/train_modified_dataset.csv"
TRAIN_LENGHTS_DATASET  = DATASET_PATH / "dataset/train_modified_dataset_lenghts.csv"
TEST_MODIFIED_DATASET  = DATASET_PATH / "dataset/test_modified_dataset.csv"
TEST_LENGHTS_DATASET   = DATASET_PATH / "dataset/test_modified_dataset_lenghts.csv"

# How many step to take for each series
SERIES_SPLIT_NUMBER = 100

# Check if CUDA is available to run things on the GPU
DEVICE = 'cpu' if not torch.cuda.is_available else 'cuda'

# Some training constants
TRAIN_PERCENTAGE = 70 # Train Percentage used to split the dataset
INITIAL_LR = 0.001    # Learning rate for optimizer
N_BATCHES  = 32       # Number of batches