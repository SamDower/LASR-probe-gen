import probe_gen.probes as probes
from probe_gen.probes import save_probe_dict_results
from probe_gen.probes.wandb_interface import load_probe_eval_dict_by_dict, load_probe_eval_dicts_as_df
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

   
def run_grid_experiment(dataset_names, layer_list, use_bias_list, normalize_list, C_list, activations_model):

    train_datasets = {}
    val_datasets = {}
    test_datasets = {}
    for dataset_name in dataset_names:
        for layer in layer_list:
            if not f"{dataset_name}_{layer}" in train_datasets:

                activations_tensor, attention_mask, labels_tensor = probes.load_hf_activations_and_labels_at_layer(dataset_name, layer, verbose=True)
                activations_tensor = probes.MeanAggregation()(activations_tensor, attention_mask)
                train_dataset, val_dataset, test_dataset = probes.create_activation_datasets(activations_tensor, labels_tensor, val_size=0.1, test_size=0.2, balance=True, verbose=True)
                train_datasets[f"{dataset_name}_{layer}"] = train_dataset
                val_datasets[f"{dataset_name}_{layer}"] = val_dataset
                test_datasets[f"{dataset_name}_{layer}"] = test_dataset


    for train_index in range(len(dataset_names)):
        train_dataset_name = dataset_names[train_index]
        # Initialise and fit a probe with the dataset
        probe = probes.SklearnLogisticProbe(use_bias=use_bias_list[train_index], normalize=normalize_list[train_index], C=C_list[train_index])
        train_set = train_datasets[f"{train_dataset_name}_{layer_list[train_index]}"]
        val_set = val_datasets[f"{train_dataset_name}_{layer_list[train_index]}"]
        probe.fit(train_set, val_set)

        for test_dataset_name in dataset_names:
            test_set = test_datasets[f"{test_dataset_name}_{layer_list[train_index]}"]
            eval_dict, _, _ = probe.eval(test_set)
            save_probe_dict_results(
                eval_dict, 
                "mean", 
                use_bias_list[train_index], 
                normalize_list[train_index], 
                C_list[train_index], 
                layer_list[train_index], 
                train_dataset_name, 
                test_dataset_name, 
                activations_model
            )

def plot_grid_experiment(dataset_names, tick_labels, layer_list, use_bias_list, normalize_list, C_list, activations_model, metric):
    """
    Plots a grid showing a metric for probes trained and tested on each of the specified datasets in a grid.

    Args:
        dataset_list (array): A list of all of the dataset names (as stored on wandb) to form the rows and columns of the grid.
        metric (str): The metric to plot in each cell of the grid (e.g. 'accuracy', 'roc_auc').
    
    Returns:
        None.
    """

    results_table = np.full((len(dataset_names), len(dataset_names)), -1, dtype=float)
    for train_index in range(len(dataset_names)):
        for test_index in range(len(dataset_names)):
            results = load_probe_eval_dict_by_dict({
                "config.train_dataset": dataset_names[train_index],
                "config.test_dataset": dataset_names[test_index],
                "config.layer": layer_list[train_index],
                "config.probe/type": "mean",
                "config.probe/use_bias": use_bias_list[train_index],
                "config.probe/normalize": normalize_list[train_index],
                "config.probe/C": C_list[train_index],
                "config.activations_model": activations_model,
                "state": "finished"  # Only completed runs
            })
            results_table[train_index, test_index] = results[metric]
            print(f"{dataset_names[train_index]}, {dataset_names[test_index]}, {results[metric]}")
    
    fig, ax = plt.subplots()

    # Create the heatmap with seaborn
    sns.heatmap(results_table, 
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            annot=True,  # This adds the text annotations
            fmt='.3f',   # Format numbers to 3 decimal places
            cmap='Greens',  # You can change the colormap
            vmin=0.5,
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
