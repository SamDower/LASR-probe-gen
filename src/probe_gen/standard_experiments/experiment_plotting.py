import wandb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from probes.wandb_interface import load_probe_eval_dict_by_datasets



def plot_results_table(dataset_list, metric):
    """
    Plots a grid showing a metric for probes trained and tested on each of the specified datasets in a grid.

    Args:
        dataset_list (array): A list of all of the dataset names (as stored on wandb) to form the rows and columns of the grid.
        metric (str): The metric to plot in each cell of the grid (e.g. 'accuracy', 'roc_auc').
    
    Returns:
        eval_dict (dict): evalualtion dictionary.
    """

    results_table = np.full((len(dataset_list), len(dataset_list)), -1, dtype=float)
    for train_index in range(len(dataset_list)):
        for test_index in range(len(dataset_list)):
            results = load_probe_eval_dict_by_datasets(dataset_list[train_index], dataset_list[test_index])
            print(results)
            print(type(results[metric]))
            results_table[train_index, test_index] = results[metric]
    
    fig, ax = plt.subplots()

    # Create the heatmap with seaborn
    sns.heatmap(results_table, 
            xticklabels=dataset_list,
            yticklabels=dataset_list,
            annot=True,  # This adds the text annotations
            fmt='.3f',   # Format numbers to 3 decimal places
            #cmap='viridis',  # You can change the colormap
            vmin=0,
            vmax=1,
            ax=ax)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")

    # Set labels and title
    plt.xlabel('Test set')
    plt.ylabel('Train set')
    ax.set_title(f"{metric}")

    fig.tight_layout()
    plt.show()