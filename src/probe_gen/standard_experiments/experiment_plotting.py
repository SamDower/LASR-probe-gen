import wandb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns




def get_eval_results_by_datasets(train_dataset, test_dataset):
    """Get accuracy for runs matching specific train and test datasets"""
    api = wandb.Api()
    
    # Query runs with specific config filters
    runs = api.runs(
        "samdower/LASR_probe_gen",
        filters={
            "config.train_dataset": train_dataset,
            "config.test_dataset": test_dataset,
            "state": "finished"  # Only completed runs
        }
    )
    
    results = []
    for run in runs:
        accuracy = run.summary.get('accuracy', None)
        roc_auc = run.summary.get('roc_auc', None)
        tpr_at_1_fpr = run.summary.get('tpr_at_1_fpr', None)
        if accuracy is not None:
            results.append({
                'run_id': run.id,
                'run_name': run.name,
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'tpr_at_1_fpr': tpr_at_1_fpr,
                'train_dataset': run.config.get('train_dataset'),
                'test_dataset': run.config.get('test_dataset')
            })
    
    if len(results) > 1:
        print(f"### WARNING ###: multiple runs for dataset pair ({train_dataset}, {test_dataset}), returning latest.")
    elif len(results) == 0:
        print(f"### WARNING ###: could not find run for dataset pair ({train_dataset}, {test_dataset}), returning None.")
        return None
    return results[-1]

def plot_results_table(dataset_list, metric):

    results_table = np.full((len(dataset_list), len(dataset_list)), -1, dtype=float)
    for train_index in range(len(dataset_list)):
        for test_index in range(len(dataset_list)):
            results = get_eval_results_by_datasets(dataset_list[train_index], dataset_list[test_index])
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