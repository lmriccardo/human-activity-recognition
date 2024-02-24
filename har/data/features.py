import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from pathlib import Path
from typing import Tuple, Dict

from har.data.visualize import (
    plot_number_of_activities_per_user,
    plot_number_of_timestamps_for_activity,
    plot_correlation_matrix,
    plot_activity_distributions,
    plot_timeseries_per_user
)


def feature_extractor(dataset_data : pd.DataFrame) -> None:
    """
    May print some useful information about the dataset

    Parameters
    ----------
    dataset_data : pd.DataFrame
        The entire dataset of data loaded as a CSV
    """
    # Take the number of users
    number_of_users = dataset_data['user'].max()

    # Take the number of activities and the list of activities
    activities = dataset_data['activity'].unique()
    number_of_activities = len(activities)

    # Take the number of activities for each user in the dataset
    compute_length_fn = lambda x : len(x)
    user_to_list_acts = dataset_data.groupby('user')['activity'].unique()
    noa_for_user = list(map(compute_length_fn, user_to_list_acts.tolist()))
    users = [ str(x) for x in range(1, number_of_users) ] 
    noa_for_user_dict = dict(zip(users, noa_for_user))

    # Looking for any NaN values
    user_na = pd.isna(dataset_data.user).sum()
    activity_na = pd.isna(dataset_data.activity).sum()
    timestamp_na = pd.isna(dataset_data.timestamp).sum()
    x_axis_na = pd.isna(dataset_data['x-axis']).sum()
    y_axis_na = pd.isna(dataset_data['y-axis']).sum()
    z_axis_na = pd.isna(dataset_data['z-axis']).sum()
    total_na = user_na + activity_na + timestamp_na + x_axis_na + y_axis_na + z_axis_na

    # Number of timestamp per activities
    not_for_acts_dict = dataset_data.activity.value_counts().to_dict()

    # Compute the correlation matrix
    # Notes that there are some features that are not numeric. In this case
    # The activity features are string, hence we cannot include them into the 
    # correlation matrix calculation.
    correlation_matrix = dataset_data.corr(numeric_only=True)

    # Compute another matrix representing for each user the number
    # of timestamps for each activities
    user_not_acts_matrix = pd.crosstab(dataset_data.user, dataset_data.activity)

    print("DATASET INFORMATION")
    print("___________________\n")
    print(f"Number of user       : {number_of_users}")
    print(f"Activities           : {', '.join(activities)}")
    print(f"Number of activities : {number_of_activities}")
    print(f"Number of NaN Values : Total count {total_na}")
    print(f"    Users     : {user_na}")
    print(f"    Activity  : {activity_na}")
    print(f"    Timestamp : {timestamp_na}")
    print(f"    X-axis    : {x_axis_na}")
    print(f"    Y-axis    : {y_axis_na}")
    print(f"    Z-axis    : {z_axis_na}")
    print()
    print("PLOTTING FEATURES")
    print("_________________\n")

    # Plotting some features
    plt.figure(figsize=(20, 3))

    # Number of activities per user
    plt.subplot(121)
    plot_number_of_activities_per_user(noa_for_user_dict)
    
    # Number of timestamp per activity
    plt.subplot(122)
    plot_number_of_timestamps_for_activity(not_for_acts_dict)
    plt.show()

    # Correlation matrix
    plt.figure(figsize=(5,5))
    plot_correlation_matrix(correlation_matrix)

    # User vs. Number of timestamp per activity
    plt.figure(figsize=(10,10))
    plot_correlation_matrix(user_not_acts_matrix, title="Users vs. Number of timestamps per Activity")

    plot_activity_distributions(dataset_data, activities)


def per_user_features_extractor(
    dataset_data : pd.DataFrame, user_value : int, save: bool=True, figure_path: str=os.getcwd()
) -> None:
    """
    Describes features from the dataset given a specific user

    Parameters
    ----------
    dataset_data : pd.DataFrame
        The entire dataset of data 
    
    user_value : int
        The value of the user in the dataset

    save : bool (default = True)
        Flag to enable saving all the images or not
    
    figure_path : str (default current path)
        Where to save all the figures
    """
    # Take the values corresponding to the input user
    df_user = dataset_data.loc[dataset_data.user == user_value].copy()
    df_user = df_user.reset_index()

    # Take all the activities
    activities = df_user.activity.unique()

    # Print informations about the mean and standard deviation of the X,Y and Z axis
    print()
    print("Standard Deviation of X,Y and Z Axis values")
    print(df_user.iloc[:, [-3, -2, -1]].std(numeric_only=True))
    print()

    print("Mean of X,Y and Z Axis values")
    print(df_user.iloc[:, [-3, -2, -1]].mean(numeric_only=True))

    print()
    plot_timeseries_per_user(user_value, df_user, activities, save=False)


def modify_feauture_dataset(
    in_data: pd.DataFrame, path_to_save: str, save: bool=False
) -> Tuple[pd.DataFrame, Dict[str, int]] | None:
    """
    Create a new dataset by modifying the input one. In this we will get rid
    of the user specification and the timestamp specification (we do not need
    them). Notice that the actual presence of which user is releated to which 
    activity it is not so important. Since we would like to classify the time serie as 
    just the activity it represents, we might just consider a single user which does all 
    the activities in different days. 
    
    Another thing that might not be important is the
    timestamp. Treating each sequence of point already consider them as time-dependent,
    hence having just a sequence of always ascending values is irrelevant to our 
    objective.

    The final dataset will have a number of columns equal to N_u x 3 x U
    where N_u is the number of activities of the user u, 3 came from the 3
    axis and U is the number of total users.

    Parameters
    ----------
    in_data : pd.DataFrame
        The original dataset after feature engineering
    
    path_to_save : str
        Where to save the final dataset if needed
    
    save : bool (default=True)
        Whether to save or not the dataset

    Returns
    -------
    pd.DataFrame | None
        It will return the final dataset whenever the user do not chose
        to save it, otherwise it will save the dataset and return nothing.
        It will also returns, in case of no saving, another dictionary
        that containsn for each activity of each user the length of the
        original timeseries, since it has been extended in the new dataset.
    """
    df_new_data = dict()
    timeseries_lengths = {"Activity" : [], "Size" : []}

    # Take the maximum length a time serie is in the original dataset
    # this will be used to pad those which are smaller
    maximum_ts_length = in_data.groupby(['user', 'activity']).count()['timestamp'].max()
    
    # Loop for all the users
    for user_value in in_data.user.unique():
        # Loop for all activity of the user
        df_user_tmp = in_data.loc[in_data.user == user_value].copy()
        df_user_tmp.reset_index(inplace=True)

        for activity in df_user_tmp.loc[:, 'activity'].unique():
            header_name_x = f"{activity.lower()}_{user_value}_X"
            header_name_y = f"{activity.lower()}_{user_value}_Y"
            header_name_z = f"{activity.lower()}_{user_value}_Z"
            
            df_user_act = df_user_tmp.loc[df_user_tmp.activity == activity] \
                                     .iloc[:, [-4, -3, -2, -1]]             \
                                     .copy()

            x_series = df_user_act['x-axis'].values
            y_series = df_user_act['y-axis'].values
            z_series = df_user_act['z-axis'].values

            timeseries_lengths["Activity"].append(f"{activity.lower()}_{user_value}")
            timeseries_lengths["Size"].append(x_series.size)

            if x_series.size < maximum_ts_length:
                padding_size = maximum_ts_length - x_series.size
                x_series = np.pad(x_series, (0, padding_size), mode='constant')
                y_series = np.pad(y_series, (0, padding_size), mode='constant')
                z_series = np.pad(z_series, (0, padding_size), mode='constant')

            df_new_data[header_name_x] = x_series
            df_new_data[header_name_y] = y_series
            df_new_data[header_name_z] = z_series
        
    df_data = pd.DataFrame(df_new_data)
    df_data_lengths = pd.DataFrame(timeseries_lengths)

    # If we do not have to save the dataset then just return it 
    if not save: return df_data, df_data_lengths

    # Otherwise we need to save it, and also the additional dictionary
    df_data.to_csv(path_to_save)

    path_to_save = path_to_save if isinstance(path_to_save, Path) else Path(path_to_save)
    df_data_name = path_to_save.stem
    path_to_save_add = path_to_save.parent / f"{df_data_name}_lenghts.csv"
    df_data_lengths.to_csv(path_to_save_add)