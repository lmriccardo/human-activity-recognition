from pathlib import Path

TRAIN_PERCENTAGE = 70 # Train Percentage on user selection

# Absolute path to the original and modified dataset
DATASET_PATH = Path(__file__).absolute().parent.parent.parent
ORIGINAL_DATASET = DATASET_PATH / "dataset/time_series_data_human_activities.csv"
MODIFIED_DATASET = DATASET_PATH / "dataset/modified_dataset.csv"
LENGHTS_DATASET  = DATASET_PATH / "dataset/modified_dataset_lenghts.csv"

