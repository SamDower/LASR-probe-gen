import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import torch

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
        None.
    """

    results_table = np.full((len(dataset_list), len(dataset_list)), -1, dtype=float)
    for train_index in range(len(dataset_list)):
        for test_index in range(len(dataset_list)):
            results = load_probe_eval_dict_by_datasets(dataset_list[train_index], dataset_list[test_index])
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
            ax=ax,
            annot_kws={"size": 20})

    # Rotate x-axis labels
    #plt.xticks(rotation=45, ha="right", rotation_mode="anchor")

    # Set labels and title
    plt.xlabel('Test set')
    plt.ylabel('Train set')
    ax.set_title(f"{metric}")

    fig.tight_layout()
    plt.show()




def plot_roc_curves(ys, y_pred_probas, labels):
    """
    Plots ROC curves for a list of true labels, predicted probabilities, and labels all on the same graph.

    Args:
        ys (array): A list of tensors of true labels, each tensor has shape [batch_size].
        y_pred_probas (array): A list of tensors of predicted probabilities, each tensor has shape [batch_size].
        labels (array): A list of strings to label each curve with.
    
    Returns:
        None.
    """

    def _safe_to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data  # Already numpy
        else:
            return np.array(data)  # Convert other types
    
    plt.figure(figsize=(8, 6))

    for i in range(len(ys)):

        y = _safe_to_numpy(ys[i])
        y_pred_proba = _safe_to_numpy(y_pred_probas[i])

        fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
        roc_auc = roc_auc_score(y, y_pred_proba)

        plt.plot(fpr, tpr, label=f'{labels[i]} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()





def plot_per_class_prediction_distributions(y, y_pred_proba):
    """
    Plots histograms for per class predicted probabilities.

    Args:
        y (tensor): A tensors of true labels with shape [batch_size].
        y_pred_proba (tensor): A tensor of predicted probabilities with shape [batch_size].
    
    Returns:
        None.
    """
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 1, 1)
    plt.hist(y_pred_proba[y.numpy() == 0], alpha=0.5, label='Negative (0)', bins=20)
    plt.hist(y_pred_proba[y.numpy() == 1], alpha=0.5, label='Positive (1)', bins=20)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Prediction Distribution')

    plt.tight_layout()
    plt.show()