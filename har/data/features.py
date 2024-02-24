import pandas as pd
import matplotlib.pyplot as plt
import os

from har.data.visualize import *


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