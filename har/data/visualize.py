import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from typing import Dict, List


def plot_number_of_activities_per_user(noa_for_user: Dict[str, int]) -> None:
    """
    Create a bar plot representing the number of activities for each user

    Parameters
    ----------
    noa_for_users : Dict[str, int]
        The number of activities for each user
    """
    plt.bar(noa_for_user.keys(), height=noa_for_user.values())
    plt.xlabel('Users (1-36)')
    plt.ylabel('Number of Activities')
    plt.title('Number of Activities per User')
    plt.grid()


def plot_number_of_timestamps_for_activity(not_for_acts: Dict[str, int]) -> None:
    """
    Create a bar plot representing the number of timestamps for each activity

    Parameters
    ----------
    not_for_acts : Dict[str, int]
        The number of timestamps for each activity
    """
    plt.bar(not_for_acts.keys(), height=not_for_acts.values())
    plt.xlabel('Activity')
    plt.ylabel('Number of Timestamps')
    plt.title('Number of Timestamps per Activity')
    plt.grid()


def plot_correlation_matrix(correlation_matrix : pd.DataFrame, title: str="Correlation Matrix") -> None:
    """
    Plot the correlation matrix using the heatmap

    Parameters
    ----------
    correlation_matrix : pd.DataFrame
        The correlation between each numeric features only
    """
    sn.heatmap(correlation_matrix, annot=True)
    plt.title(title)
    plt.show()


def plot_activity_distributions(dataset_data: pd.DataFrame, activities: List[str]) -> None:
    """
    Plot the distribution of X, Y and Z axes for each activity in the dataset

    Parameters
    ----------
    dataset_data : pd.DataFrame
        The entire dataset
    
    activities : List[str]
        The list of activities in the dataset
    """
    n_bins = 100

    for activity in activities:
        df_act = dataset_data.loc[dataset_data.activity == activity].copy()
        df_act.reset_index(inplace=True)

        plt.figure(figsize=(20, 3))

        plt.subplot(131)
        plt.hist(df_act['x-axis'], n_bins, color='red', alpha=0.5)
        plt.xlim(-20, 20)
        plt.title(f'{activity} - X Axis')
        plt.grid()

        plt.subplot(132)
        plt.hist(df_act['y-axis'], n_bins, color='green', alpha=0.5)
        plt.xlim(-20, 20)
        plt.title(f'{activity} - Y Axis')
        plt.grid()

        plt.subplot(133)
        plt.hist(df_act['z-axis'], n_bins, color='blue', alpha=0.5)
        plt.xlim(-20, 20)
        plt.title(f'{activity} - Z Axis')
        plt.grid()

        plt.show()


def plot_timeseries_per_user(
    user_name: int, df_user : pd.DataFrame, activities: List[str], 
    figure_path: str, save: bool=False
) -> None:
    """
    Plot the timeseries for each activity for each axis for a single user

    Parameters
    ----------
    user_name : int
        The name of the user

    df_user : pd.DataFrame
        The dataset for a single user

    activities : List[str]
        The list of all activities in the dataset

    save : bool (default=True)
        Flag to enable saving the figure or not
    
    figure_path : str (default current path)
        Where to save all the figures
    """
    for activity in activities:
        df_temp = df_user.loc[df_user.activity == activity].copy()
        df_temp.reset_index(inplace=True)

        # Convert time to seconds and start with 0
        t_minutes = df_temp.timestamp.min()
        df_temp['time_sec'] = (df_temp['timestamp'] - t_minutes) / 1.0e9

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14,4), sharex=True)
        fig.tight_layout()
    
        ax1.plot(df_temp['time_sec'], df_temp['x-axis'], color='red', alpha=0.5)
        ax1.set_ylim(-20,20)
        ax1.set_title(f"Activity : {activity} - X Axis")
        ax1.set_ylabel('Accelerations')
        ax1.grid()
        
        ax2.plot(df_temp['time_sec'], df_temp['y-axis'], color='green', alpha=0.5)
        ax2.set_ylim(-20,20)
        ax2.set_title(f"Activity : {activity} - Y Axis")
        ax2.set_ylabel('Accelerations')
        ax2.grid()

        ax3.plot(df_temp['time_sec'], df_temp['z-axis'], color='blue', alpha=0.5)
        ax3.set_ylim(-20,20)    
        ax3.set_title(f"Activity : {activity} - Z Axis")
        ax3.set_ylabel('Accelerations')
        ax3.grid()
        
        plt.xlabel('Time [seconds]')
        plt.show()

        if save:
            figname = f"user{user_name}_{activity.lower()}_timeseries_xyz.png"
            plt.savefig(figure_path / figname)