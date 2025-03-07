import matplotlib.pyplot as plt
import pandas as pd


def plot_histograms(df, n_cols):
    """
    A function to create histogram subplots for each column of a pandas dataframe.

    Args:
        df (pd.DataFrame): dataframe to plot histogram of
        n_cols (int): number of =subplot columns
    """
    num_columns = df.shape[1]
    
    # Determine the number of rows and columns for subplots
    num_rows = (num_columns + 1) // n_cols
    num_cols = n_cols 
    
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5))
    axs = axs.flatten()

    # For each colum
    for i, column in enumerate(df.columns):
        df[column].sort_values().hist(ax=axs[i])
        axs[i].set_title(f"Histogram of {column}")
    
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    
    
def plot_scatters(df, n_cols,target):
    """
    A function to create scatter subplots for each column of a pandas dataframe.

    Args:
        df (pd.DataFrame): dataframe to plot histogram of
        n_cols (int): number of =subplot columns
        target (string): name of target variable column
    """
    num_columns = df.shape[1]
    
    # Determine the number of rows and columns for subplots
    num_rows = (num_columns + 1) // n_cols
    num_cols = n_cols 
    
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5))
    axs = axs.flatten()

    # For each colum
    for i, column in enumerate(df.columns):
        df.plot.scatter(x=column, y=target, ax=axs[i])
        axs[i].set_title(f"Scatterplot of {column}")
    
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()