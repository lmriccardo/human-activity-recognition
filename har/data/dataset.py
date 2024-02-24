import torch
import torch.utils.data as torchdata
import pandas as pd
import math

from har.utils.constants import TRAIN_PERCENTAGE, SERIES_SPLIT_NUMBER
from har.data.features import modify_feauture_dataset
from typing import Tuple, List
from pathlib import Path


class HumanActivityRecognitionDataset(torchdata.Dataset):
    """**Human Activity Recognition Task Dataset**
    
    Attributes
    ----------
    ...
    """
    def __init__(
        self, ds_path: Path |str | None=None, ds_path_lengths: Path |str | None=None,
        ds_data : pd.DataFrame | None=None, ds_data_lengths : pd.DataFrame | None=None,
        series_split_n : int=SERIES_SPLIT_NUMBER
    ) -> None:
        self.data = ds_data
        self.data_size = ds_data_lengths

        # If both ds_path and ds_path_lengths 
        if ds_path is not None and ds_path_lengths is not None:
            self.data = pd.read_csv(ds_path)
            self.data_size = pd.read_csv(ds_path_lengths)

        # Check that the two dataset are not None
        assert self.data is not None and self.data_size is not None, \
            "Error: The input dataset is empty or None. \n" + \
            "Please provide a dataset either already loaded from the CSV " + \
            "file or provide at least the two CSV files"
        
        self.series_split_n = series_split_n
        self.labels = list(set([ x.split("_")[0] for x in self.data_size.Activity ]))

    def __len__(self) -> int:
        """ Return the length of the dataset """
        total_size = 0
        for index in range(self.data_size.shape[0]):
            size = self.data_size.iloc[index, -1]
            total_size = total_size + size // self.series_split_n

        return total_size
    
    def divide_data(self) -> List[Tuple[int, Tuple[List[float], List[float], List[float]]]]:
        resulting_data = []
        for activity in self.data_size.Activity:
            data_size = self.data_size.loc[self.data_size.Activity == activity] \
                            .copy()                                             \
                            .Size                                               \
                            .values[0]
            
            activity_index_label = self.labels.index(activity.split("_")[0])

            x_series = self.data[f"{activity}_X"].values[:data_size]
            y_series = self.data[f"{activity}_Y"].values[:data_size]
            z_series = self.data[f"{activity}_Z"].values[:data_size]

            n_subseries = data_size // self.series_split_n
            for subindex in range(0, n_subseries):
                x_subserie = x_series[subindex * n_subseries : (subindex + 1) * n_subseries]
                y_subserie = y_series[subindex * n_subseries : (subindex + 1) * n_subseries]
                z_subserie = z_series[subindex * n_subseries : (subindex + 1) * n_subseries]
                resulting_data.append(
                    (activity_index_label, (x_subserie, y_subserie, z_subserie))
                )
        
        return resulting_data
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx > self.__len__():
            raise ValueError(
                f"Index value {idx} is bigger then the dataset size"
            )

        label, (x, y, z) = self.final_data[idx]

        x_tensor = torch.tensor(x, dtype=torch.float64, device='cpu')
        y_tensor = torch.tensor(y, dtype=torch.float64, device='cpu')
        z_tensor = torch.tensor(z, dtype=torch.float64, device='cpu')

        return (label, torch.vstack((x_tensor, y_tensor, z_tensor)))
        

def split_dataset(
    dataset: Tuple[pd.DataFrame, pd.DataFrame | None], train_perc: int=TRAIN_PERCENTAGE
) -> Tuple[HumanActivityRecognitionDataset, HumanActivityRecognitionDataset]:
    """
    Divide the dataset into train and test. Notice that the dataset can be either
    the one not being modified by the feature module, or the modified one. In the
    first case only the first element of the tuple will not be None. If this is the
    case then the modify_feature_dataset will be called. On the other hand, if both
    values are not None, then the modification will not be done.

    Parameters
    ----------
    dataset: Tuple[pd.DataFrame, pd.DataFrame | None]
        The dataset that can be either the original dataset (only one entry),
        or the modified one (with both entries).

    train_perc : float
        The percentage of training test to consider

    Returns
    -------
    Tuple[HumanActivityRecognitionDataset, HumanActivityRecognitionDataset]
        A tuple of Dataset object respectively for train and test
    """
    df_data, df_data_lengths = dataset

    # Check if the second entry is None, in this case apply modification
    if not df_data_lengths:
        df_data, df_data_lengths = modify_feauture_dataset(df_data, None, False)
    
    train_size = math.ceil(df_data_lengths.shape[0] * train_perc / 100)

    # Take the train set
    df_train = df_data.iloc[:, 0:3 * (train_size + 1)]
    df_train_lengths = df_data_lengths.iloc[0:train_size, :]
    train = HumanActivityRecognitionDataset(ds_data=df_train, ds_data_lengths=df_train_lengths)

    # Take the test set
    df_test = df_data.iloc[:, 3 * (train_size + 1):]
    df_test_lengths = df_data_lengths.iloc[train_size:, :]
    test = HumanActivityRecognitionDataset(ds_data=df_test, ds_data_lengths=df_test_lengths)

    return train, test