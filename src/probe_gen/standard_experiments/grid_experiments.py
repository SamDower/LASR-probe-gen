import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

import probe_gen.probes as probes
from probe_gen.paths import data
from probe_gen.probes.wandb_interface import load_probe_eval_dict_by_dict
from probe_gen.standard_experiments.hyperparameter_search import (
    get_best_hyperparams_for_train_setup,
)


def get_grid_results_table_from_wandb(train_setup, test_setup, metric="roc_auc"):
    """
    Gets the results table from wandb for the probes specified in the train_setup and test_setup lists.
    Args:
        train_setup (list): 
            [probe_type, behaviour, datasource, activations_model, generation_method, response_model, mode, cfg]
        test_setup (list): 
            [behaviour, datasource, activations_model, generation_method, response_model, mode]
        metric (str): The metric to plot in each cell of the grid (e.g. 'accuracy', 'roc_auc').
    """
    tr = train_setup
    ts = test_setup
    
    results_table = np.full((len(tr), len(ts)), -1, dtype=float)
    for i in range(len(tr)):
        cfg = tr[i][7]
        for j in range(len(ts)):
            search_dict = {
                "config.probe/type": tr[i][0],
                "config.behaviour": tr[i][1],
                "config.train/datasource": tr[i][2],
                "config.train/generation_method": tr[i][4],
                "config.train/response_model": tr[i][5],
                "config.test/datasource": ts[j][1],
                "config.test/generation_method": ts[j][3],
                "config.test/response_model": ts[j][4],
                "config.layer": cfg.layer,
                "config.probe/use_bias": cfg.use_bias,
                "config.probe/normalize": cfg.normalize,
                "config.activations_model": tr[i][3],
                "state": "finished"  # Only completed runs
            }
            if "torch" in tr[i][0]:
                search_dict["config.probe/lr"] = cfg.lr
                search_dict["config.probe/weight_decay"] = cfg.weight_decay
            elif tr[i][0] == "mean":
                search_dict["config.probe/C"] = cfg.C
            results = load_probe_eval_dict_by_dict(search_dict)
            results_table[i, j] = results[metric]
            # print(f"{train_dataset_name}, {test_dataset_names[j]}, {results[metric]}")
    return results_table


def plot_grid_experiment_lean(train_setup, test_setup, metric="roc_auc"):
    """
    Plots a grid experiment on the probes specified in the probes_setup list.
    Args:
        train_setup (list): 
            [probe_type, behaviour, datasource, activations_model, generation_method, response_model, mode, cfg]
        test_setup (list): 
            [behaviour, datasource, activations_model, generation_method, response_model, mode]
        metric (str): The metric to plot in each cell of the grid (e.g. 'accuracy', 'roc_auc').
    """
    tr = train_setup
    ts = test_setup
    
    # Get the best hyperparameters for each probe if not provided
    tr = get_best_hyperparams_for_train_setup(tr)
    
    # Get all results by querying wandb for all run configs
    results_table = get_grid_results_table_from_wandb(tr, ts, metric)

    # Get tick labels
    train_labels = [tr[i][5]+"_"+tr[i][4] for i in range(len(tr))]
    test_labels = [ts[j][4]+"_"+ts[j][3] for j in range(len(ts))]

    # Create the heatmap with seaborn
    fig, ax = plt.subplots()
    sns.heatmap(
        results_table,
        xticklabels=test_labels,
        yticklabels=train_labels,
        annot=True,  # This adds the text annotations
        fmt=".3f",  # Format numbers to 3 decimal places
        cmap="Greens",  # You can change the colormap
        vmin=0.5,
        vmax=1,
        ax=ax,
        annot_kws={"size": 12}, # change this to 12 for 6x6 grids
    )

    # Rotate x-axis labels
    # plt.xticks(rotation=45, ha="right", rotation_mode="anchor")

    # Set labels and title
    plt.xlabel("Test set")
    plt.ylabel("Train set")
    ax.set_title(f"{metric}")

    fig.tight_layout()
    plt.show()


def plot_grid_experiment_lean_with_means(train_setup, test_setup, metric="roc_auc", save=False, min_metric=0.5, max_metric=1):
    """
    Plots a grid experiment on the probes specified in the probes_setup list.
    Args:
        train_setup (list): 
            [probe_type, behaviour, datasource, activations_model, generation_method, response_model, mode, cfg]
        test_setup (list): 
            [behaviour, datasource, activations_model, generation_method, response_model, mode]
        metric (str): The metric to plot in each cell of the grid (e.g. 'accuracy', 'roc_auc').
        save (bool): Whether to save the figure.
        min_metric (float): The minimum metric value to plot.
        max_metric (float): The maximum metric value to plot.
    """
    tr = train_setup
    ts = test_setup
    
    # Get the best hyperparameters for each probe if not provided
    tr = get_best_hyperparams_for_train_setup(tr)
    
    # Get all results by querying wandb for all run configs
    results_table = get_grid_results_table_from_wandb(tr, ts, metric)

    # Get tick labels
    abridge = lambda label: "".join([p[0] for p in label.split("_") if p])  # e.g., llama_3b → l3b
    train_full_labels = [tr[i][5]+"_"+tr[i][4] for i in range(len(tr))]
    test_full_labels = [ts[j][4]+"_"+ts[j][3] for j in range(len(ts))]
    train_short_labels = [abridge(lbl) for lbl in train_full_labels]
    test_short_labels = [abridge(lbl) for lbl in test_full_labels]

    # Add Row and Column Means
    row_means = np.mean(results_table, axis=1, keepdims=True)
    col_means = np.mean(results_table, axis=0, keepdims=True)
    full_table = np.block([
        [results_table, row_means],
        [col_means, np.array([[np.nan]])],
    ])
    mask = np.isnan(full_table)

    # Create the heatmap with seaborn
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use min/max of valid entries if not provided
    valid_values = results_table[results_table != -1]
    min_metric = min_metric if min_metric is not None else (np.min(valid_values) if valid_values.size > 0 else 0)
    max_metric = max_metric if max_metric is not None else (np.max(valid_values) if valid_values.size > 0 else 1)

    sns.heatmap(
        full_table,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="Greens",
        vmin=min_metric,
        vmax=max_metric,
        cbar=True,
        ax=ax,
        linewidths=0,  # no grid between normal cells
        linecolor='white',
        annot_kws={"size": 12},
        xticklabels=test_short_labels + ["Mean"],
        yticklabels=train_short_labels + ["Mean"],
    )

    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    # Draw separating lines between main grid and means
    n_rows, n_cols = results_table.shape
    ax.axhline(n_rows, color='white', linewidth=2)
    ax.axvline(n_cols, color='white', linewidth=2)

    # Legend for abbreviations
    legend_elements = []
    for short, full in zip(test_short_labels, test_full_labels):
        legend_elements.append(Patch(facecolor='none', edgecolor='none', label=rf"$\mathbf{{{short}}}$: {full}"))
    for short, full in zip(train_short_labels, train_full_labels):
        legend_elements.append(Patch(facecolor='none', edgecolor='none', label=rf"$\mathbf{{{short}}}$: {full}"))
    ax.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1.15, 0.5),
        title="",
        frameon=False
    )

    ax.set_xlabel("Test", fontsize=12, fontweight="bold")
    ax.set_ylabel("Train", fontsize=12, fontweight="bold")
    ax.set_title(f"{behaviour}, {metric}", fontsize=14, fontweight="bold")
    fig.tight_layout()

    # Save the figure
    if save:
        save_path = data.figures / behaviour / f"{behaviour}_{metric}_heatmap.png"
        plt.savefig(save_path, dpi=300)
        plt.savefig(save_path.path.with_suffix(".pdf"), dpi=300)
    plt.show()
    

def run_grid_experiment_lean(train_setup, test_setup):
    """
    Runs a grid experiment on the probes specified in the train_setup list.
    Args:
        train_setup (list): 
            [probe_type, behaviour, datasource, activations_model, generation_method, response_model, mode, cfg]
        test_setup (list): 
            [behaviour, datasource, activations_model, generation_method, response_model, mode]
    """
    tr = train_setup
    ts = test_setup
    
    # Get the best hyperparameters for each probe if not provided
    tr = get_best_hyperparams_for_train_setup(tr)

    for i in tqdm(range(len(tr))):
        cfg = tr[i][7]
        
        # Get train and val datasets
        activations_tensor, attention_mask, labels_tensor = probes.load_hf_activations_at_layer(
            tr[i][1], tr[i][2], tr[i][3], tr[i][5], tr[i][4], tr[i][6], cfg.layer, and_labels=True, verbose=False)
        if "mean" in tr[i][0]:
            activations_tensor = probes.MeanAggregation()(activations_tensor, attention_mask)
        split_val = 2500 if tr[i][1] in ["deception", "sandbagging"] else 3500
        split_val = 3000 if tr[i][2] == "shakespeare" else split_val
        train_dataset, val_dataset, _ = probes.create_activation_datasets(activations_tensor, labels_tensor, splits=[split_val, 500, 0])
        
        # Train the probe
        if tr[i][0] == "attention_torch":
            probe = probes.TorchAttentionProbe(cfg)
        elif tr[i][0] == "mean_torch":
            probe = probes.TorchLinearProbe(cfg)
        elif tr[i][0] == "mean":
            probe = probes.SklearnLogisticProbe(cfg)
        probe.fit(train_dataset, val_dataset)

        for j in tqdm(range(len(ts))):
            # Get test datasets, needing different layers and types for different probes
            activations_tensor, attention_mask, labels_tensor = probes.load_hf_activations_at_layer(
                ts[j][0], ts[j][1], ts[j][2], ts[j][4], ts[j][3], ts[j][5], cfg.layer, and_labels=True, verbose=False)
            if "mean" in tr[i][0]:
                activations_tensor = probes.MeanAggregation()(activations_tensor, attention_mask)
            
            # TODO: Make sure this is correct
            if ts[j][1] == "trading":
                _, _, test_dataset = probes.create_activation_datasets(activations_tensor, labels_tensor, splits=[2500, 500, 500])
            elif ts[j][0] in ["deception", "sandbagging"]:
                _, _, test_dataset = probes.create_activation_datasets(activations_tensor, labels_tensor, splits=[0, 0, 500])
            else:
                _, _, test_dataset = probes.create_activation_datasets(activations_tensor, labels_tensor, splits=[0, 0, 1000])
            
            # Evaluate the probe
            eval_dict, _, _ = probe.eval(test_dataset)
            
            # Save the results
            if "torch" in tr[i][0]:
                hyperparams = [cfg.layer, cfg.use_bias, cfg.normalize, cfg.lr, cfg.weight_decay]
            elif tr[i][0] == "mean":
                hyperparams = [cfg.layer, cfg.use_bias, cfg.normalize, cfg.C]
            probes.wandb_interface.save_probe_dict_results(
                eval_dict=eval_dict,
                probe_type = tr[i][0],
                behaviour = tr[i][1],
                train_set=[tr[i][2], tr[i][4], tr[i][5]],
                test_set=[ts[j][1], ts[j][3], ts[j][4]],
                activations_model=tr[i][3],
                hyperparams=hyperparams,
            )
