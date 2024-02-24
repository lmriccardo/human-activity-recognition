from pathlib import Path

TRAIN_PERCENTAGE = 70 # Train Percentage on user selection

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