import torch
import torch.utils.data as torchdata
import pandas as pd
import math

from har.utils.constants import TRAIN_PERCENTAGE, SERIES_SPLIT_NUMBER
from har.data.features import modify_feauture_dataset
from typing import Tuple, List
from pathlib import Path


class HumanActivityRecognitionDataset(torchdata.Dataset):
    def __init__(
        self, ds_path: Path |str | None=None,
              ds_path_lengths: Path |str | None=None,
              ds_data : pd.DataFrame | None=None,
              ds_data_lengths : pd.DataFrame | None=None,
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

        self.final_data = self.divide_data()

    @property
    def number_of_user(self) -> int:
        """ Returns the number of users taken for this dataset """
        end_user = int(self.data_size.Activity.values[-1].split("_")[1])
        start_user = int(self.data_size.Activity.values[0].split("_")[1])
        return end_user - start_user

    def divide_data(self) -> List[Tuple[int, Tuple[List[float], List[float], List[float]]]]:
        """
        Divide the input data into smaller dimensional sequences in order to have
        more data to train the neural network. The division is based on the
        parameter `series_split_n` which precise the number of elements for each
        split, and the number of split is computed as `size // series_split_n` + 1
        if `size % series_split_n` is different from 0. For example, from a series
        of 12861 data, we can create 129 subseries each of length 100. 

        Returns
        -------
        List[Tuple[int, Tuple[List[float], List[float], List[float]]]]
            The final list. At each position in the list there is a tuple:
            in the first position the classification of that part of the serie,
            in the second position another tuple with (x, y, z) subseries.
        """
        resulting_data = []
        for activity, data_size in zip(self.data_size.Activity, self.data_size.Size):
            activity_index_label = self.labels.index(activity.split("_")[0])

            x_series = self.data[f"{activity}_X"].values[:data_size]
            y_series = self.data[f"{activity}_Y"].values[:data_size]
            z_series = self.data[f"{activity}_Z"].values[:data_size]

            n_subseries = data_size // self.series_split_n
            for subindex in range(0, n_subseries + 1):
                start_idx = subindex * self.series_split_n
                end_idx = (subindex + 1) * self.series_split_n

                if subindex == n_subseries and data_size % self.series_split_n:
                    x_subserie = x_series[start_idx :]
                    y_subserie = y_series[start_idx :]
                    z_subserie = z_series[start_idx :]
                    resulting_data.append(
                        (activity_index_label, (x_subserie, y_subserie, z_subserie))
                    )
                    continue

                x_subserie = x_series[start_idx : end_idx]
                y_subserie = y_series[start_idx : end_idx]
                z_subserie = z_series[start_idx : end_idx]

                resulting_data.append(
                    (activity_index_label, (x_subserie, y_subserie, z_subserie))
                )

        return resulting_data

    def __len__(self) -> int:
        total_size = 0
        for index in range(self.data_size.shape[0]):
            size = self.data_size.iloc[index, -1]
            total_size = total_size + size // self.series_split_n
            if size % self.series_split_n != 0:
                total_size = total_size + 1

        return total_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx > self.__len__():
            raise ValueError(
                f"Index value {idx} is bigger then the dataset size"
            )

        label, (x, y, z) = self.final_data[idx]

        device = 'cpu' if not torch.cuda.is_available else 'cuda'

        x_tensor = torch.tensor(x, dtype=torch.float64, device=device)
        y_tensor = torch.tensor(y, dtype=torch.float64, device=device)
        z_tensor = torch.tensor(z, dtype=torch.float64, device=device)

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