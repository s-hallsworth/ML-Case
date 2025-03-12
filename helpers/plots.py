import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_histograms(df, n_cols):
    """
    A function to create histogram subplots for each column of a pandas dataframe,

    Args:
        df (pd.DataFrame): Dataframe to plot histograms of.
        n_cols (int): Number of subplot columns.
        target (str): Column name of target
    """
    feature_columns = df.columns
    num_features = len(feature_columns)

    # Determine number of rows needed
    num_rows = (num_features + n_cols - 1) // n_cols
    
    fig, axs = plt.subplots(num_rows, n_cols, figsize=(14, num_rows * 4))
    axs = axs.flatten()

    # Plot each feature column
    for i, column in enumerate(feature_columns):
        df[column].sort_values().hist(ax=axs[i])
        axs[i].set_title(f"Histogram of {column} ")
        axs[i].set_ylabel("Count")
        axs[i].set_xlabel(column)

    # Hide extra subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()
    
    
def plot_histograms_percent(df, n_cols, target):
    """
    A function to create histogram subplots for each column of a pandas dataframe,
    excluding the target column. The histogram bars represent the percentage makeup
    of each target class in each bin.

    Args:
        df (pd.DataFrame): Dataframe to plot histograms of.
        n_cols (int): Number of subplot columns.
        target (str): Column name of target
    """
    feature_columns = [col for col in df.columns if col != target]
    num_features = len(feature_columns)

    # Determine number of rows needed
    num_rows = (num_features + n_cols - 1) // n_cols
    
    fig, axs = plt.subplots(num_rows, n_cols, figsize=(20, num_rows * 5))
    axs = axs.flatten()

    # Plot each feature column
    for i, column in enumerate(feature_columns):
        df = df.sort_values(by=[target,column]) # Sort dataframe
        
        # Compute histogram with percentages
        sns.histplot(data=df, x=column, hue=target, ax=axs[i], bins=30, 
                     multiple="fill", kde=False, stat="percent")
        axs[i].set_title(f"Percentage Histogram of {column} (colored by {target})")
        axs[i].set_ylabel("Percentage (%)")
        axs[i].set_xlabel(column)

    # Hide extra subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

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