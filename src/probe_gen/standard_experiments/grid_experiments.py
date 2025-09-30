import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

import probe_gen.probes as probes
from probe_gen.config import ConfigDict
from probe_gen.paths import data
from probe_gen.probes.wandb_interface import load_probe_eval_dict_by_dict
from probe_gen.standard_experiments.hyperparameter_search import (
    load_best_params_from_search,
)


def run_grid_experiment_lean(train_setup, test_setup):
    """
    Runs a grid experiment on the probes specified in the probes_setup list.
    Args:
        train_setup (list): 
            [probe_type, behaviour, datasource, activations_model, generation_method, response_model, mode, cfg]
        test_setup (list): 
            [behaviour, datasource, activations_model, generation_method, response_model, mode]
    """
    activations_model = test_setup[0][3] # Assuming activations model is the same for all test setups
    
    # Get the best hyperparameters for each probe if not provided
    ps = train_setup
    for i in range(len(train_setup)):
        if not isinstance(ps[i][-1], ConfigDict):
            if ps[i][0] == 'mean':
                best_cfg = ConfigDict.from_json(ps[i][0], ps[i][1])
                ps[i] = [ps[i][0], ps[i][1], ConfigDict(layer=best_cfg.layer, use_bias=True, normalize=True, C=best_cfg.C)]
                print(ConfigDict(layer=best_cfg.layer, use_bias=True, normalize=True, C=best_cfg.C))
            else:
                best_cfg = None
                try:
                    best_cfg = ConfigDict.from_json(ps[i][0], ps[i][1])
                except KeyError:
                    print(f"No best hyperparameters found for {ps[i][0]}, {ps[i][1]} locally, pulling from wandb...")
                    best_cfg = load_best_params_from_search(ps[i][0], ps[i][1], "llama_3b")
                if best_cfg is None:
                    raise ValueError(f"No best hyperparameters found for {ps[i][0]}, {ps[i][1]}")
                ps[i] = [ps[i][0], ps[i][1], ConfigDict(best_cfg)]

    for i in tqdm(range(len(train_setup))):
        probe_type = ps[i][0]
        train_dataset_name = ps[i][1]
        cfg = ps[i][2]
        
        # Get train and val datasets
        activations_tensor, attention_mask, labels_tensor = probes.load_hf_activations_and_labels_at_layer(train_dataset_name, cfg.layer)
        if "mean" in probe_type:
            activations_tensor = probes.MeanAggregation()(activations_tensor, attention_mask)
        if "3k" in train_dataset_name or "3.5k" in train_dataset_name:
            train_dataset, val_dataset, _ = probes.create_activation_datasets(activations_tensor, labels_tensor, splits=[2500, 500, 0])
        else:
            train_dataset, val_dataset, _ = probes.create_activation_datasets(activations_tensor, labels_tensor, splits=[3500, 500, 0])
        
        # Train the probe
        if probe_type == "attention_torch":
            probe = probes.TorchAttentionProbe(cfg)
        elif probe_type == "mean_torch":
            probe = probes.TorchLinearProbe(cfg)
        elif probe_type == "mean":
            probe = probes.SklearnLogisticProbe(cfg)
        probe.fit(train_dataset, val_dataset)

        for test_dataset_name in test_setup:
            # Get test datasets, needing different layers and types for different probes
            activations_tensor, attention_mask, labels_tensor = probes.load_hf_activations_and_labels_at_layer(test_dataset_name, cfg.layer)
            if probe_type == "mean":
                activations_tensor = probes.MeanAggregation()(activations_tensor, attention_mask)
            if test_dataset_name == "jailbreaks_llama_3b_5k":
                _, _, test_dataset = probes.create_activation_datasets(activations_tensor, labels_tensor, splits=[3500, 500, 1000])
            elif "3.5k" in test_dataset_name:
                _, _, test_dataset = probes.create_activation_datasets(activations_tensor, labels_tensor, splits=[2500, 500, 500])
            elif "500" in test_dataset_name:
                _, _, test_dataset = probes.create_activation_datasets(activations_tensor, labels_tensor, splits=[0, 0, 500])
            else:
                _, _, test_dataset = probes.create_activation_datasets(activations_tensor, labels_tensor, splits=[0, 0, 1000])
            
            # Evaluate the probe
            eval_dict, _, _ = probe.eval(test_dataset)
            
            # Save the results
            if "torch" in probe_type:
                hyperparams = [cfg.layer, cfg.use_bias, cfg.normalize, cfg.lr, cfg.weight_decay]
            elif probe_type == "mean":
                hyperparams = [cfg.layer, cfg.use_bias, cfg.normalize, cfg.C]
            probes.wandb_interface.save_probe_dict_results(
                eval_dict=eval_dict, 
                train_set_name=train_dataset_name,
                test_set_name=test_dataset_name,
                activations_model=activations_model,
                probe_type=probe_type,
                hyperparams=hyperparams,
            )


def plot_grid_experiment_lean(probes_setup, test_dataset_names, activations_model, metric="roc_auc"):
    # Get the best hyperparameters for each probe if not provided    
    ps = probes_setup
    for i in range(len(probes_setup)):
        if len(ps[i]) == 2:
            if ps[i][0] == 'mean':
                best_cfg = ConfigDict.from_json(ps[i][0], ps[i][1].split("_")[0])
                print(best_cfg)
                ps[i] = [ps[i][0], ps[i][1], ConfigDict(layer=best_cfg.layer, use_bias=True, normalize=True, C=best_cfg.C)]
            else:
                best_cfg = None
                try:
                    best_cfg = ConfigDict.from_json(ps[i][0], ps[i][1])
                except KeyError:
                    print(f"No best hyperparameters found for {ps[i][0]}, {ps[i][1]} locally, pulling from wandb...")
                    best_cfg = load_best_params_from_search(ps[i][0], ps[i][1], "llama_3b")
                if best_cfg is None:
                    raise ValueError(f"No best hyperparameters found for {ps[i][0]}, {ps[i][1]}")
                ps[i] = [ps[i][0], ps[i][1], ConfigDict(best_cfg)]
    
    # Get all results by querying wandb for all run configs
    results_table = np.full((len(probes_setup), len(test_dataset_names)), -1, dtype=float)
    for i in range(len(probes_setup)):
        probe_type = ps[i][0]
        train_dataset_name = ps[i][1]
        cfg = ps[i][2]
        for j in range(len(test_dataset_names)):
            search_dict = {
                "config.probe/type": probe_type,
                "config.train_dataset": train_dataset_name,
                "config.test_dataset": test_dataset_names[j],
                "config.layer": cfg.layer,
                "config.probe/use_bias": cfg.use_bias,
                "config.probe/normalize": cfg.normalize,
                "config.activations_model": activations_model,
                "state": "finished",  # Only completed runs
            }
            if "torch" in probe_type:
                search_dict["config.probe/lr"] = cfg.lr
                search_dict["config.probe/weight_decay"] = cfg.weight_decay
            elif probe_type == "mean":
                search_dict["config.probe/C"] = cfg.C
            results = load_probe_eval_dict_by_dict(search_dict)
            results_table[i, j] = results[metric]
            # print(f"{train_dataset_name}, {test_dataset_names[j]}, {results[metric]}")

    # Get tick labels
    train_labels = [ps[i][1] for i in range(len(ps))]
    for i in range(len(train_labels)):
        train_labels[i] = train_labels[i].split("_")[1:-1]
        train_labels[i] = "_".join(train_labels[i])[:30]
    test_labels = test_dataset_names
    for i in range(len(test_labels)):
        test_labels[i] = test_labels[i].split("_")[1:-1]
        test_labels[i] = "_".join(test_labels[i])

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


## SUMMARY GRAPHS
def plot_grid_experiment_lean_with_means(probes_setup, test_dataset_names, activations_model, min_metric=None, max_metric=None, metric="roc_auc", behaviour="", save=False):
    # === Step 1: Preprocessing Setup (EXACT COPY from plot_grid_experiment_lean) ===
    ps = probes_setup
    for i in range(len(probes_setup)):
        if len(ps[i]) == 2:
            if ps[i][0] == 'mean':
                best_cfg = ConfigDict.from_json(ps[i][0], ps[i][1].split("_")[0])
                ps[i] = [ps[i][0], ps[i][1], ConfigDict(layer=best_cfg.layer, use_bias=True, normalize=True, C=best_cfg.C)]
            else:
                best_cfg = None
                try:
                    best_cfg = ConfigDict.from_json(ps[i][0], ps[i][1])
                except KeyError:
                    print(f"No best hyperparameters found for {ps[i][0]}, {ps[i][1]} locally, pulling from wandb...")
                    best_cfg = load_best_params_from_search(ps[i][0], ps[i][1], "llama_3b")
                if best_cfg is None:
                    raise ValueError(f"No best hyperparameters found for {ps[i][0]}, {ps[i][1]}")
                ps[i] = [ps[i][0], ps[i][1], ConfigDict(best_cfg)]

    # === Step 2: Collect Result Table ===
    results_table = np.full((len(probes_setup), len(test_dataset_names)), -1, dtype=float)
    for i in range(len(probes_setup)):
        probe_type = ps[i][0]
        train_dataset_name = ps[i][1]
        cfg = ps[i][2]
        for j in range(len(test_dataset_names)):
            search_dict = {
                "config.probe/type": probe_type,
                "config.train_dataset": train_dataset_name,
                "config.test_dataset": test_dataset_names[j],
                "config.layer": cfg.layer,
                "config.probe/use_bias": cfg.use_bias,
                "config.probe/normalize": cfg.normalize,
                "config.activations_model": activations_model,
                "state": "finished",
            }
            if "torch" in probe_type:
                search_dict["config.probe/lr"] = cfg.lr
                search_dict["config.probe/weight_decay"] = cfg.weight_decay
            elif probe_type == "mean":
                search_dict["config.probe/C"] = cfg.C
            results = load_probe_eval_dict_by_dict(search_dict)
            results_table[i, j] = results[metric]

    # === Step 3: Label Processing ===
    def abridge(label):
        # You can modify this logic as needed
        parts = label.split("_")
        return "".join([p[0] for p in parts if p])  # e.g., llama_3b → l3b

    train_full_labels = ["_".join(ps[i][1].split("_")[1:-1]) for i in range(len(ps))]
    test_full_labels = ["_".join(name.split("_")[1:-1]) for name in test_dataset_names]
    train_short_labels = [abridge(lbl) for lbl in train_full_labels]
    test_short_labels = [abridge(lbl) for lbl in test_full_labels]

    # === Step 4: Add Row and Column Means ===
    row_means = np.mean(results_table, axis=1, keepdims=True)
    col_means = np.mean(results_table, axis=0, keepdims=True)
    full_table = np.block([
        [results_table, row_means],
        [col_means, np.array([[np.nan]])],
    ])

    # === Step 5: Create Mask for bottom-right NaN ===
    mask = np.isnan(full_table)

    # === Step 6: Heatmap ===
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

    # === Step 7: Draw separating lines between main grid and means ===
    n_rows, n_cols = results_table.shape
    ax.axhline(n_rows, color='white', linewidth=2)
    ax.axvline(n_cols, color='white', linewidth=2)

    # === Step 8: Legend for abbreviations ===
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

    if save:
        save_path = data.figures / behaviour / f"{behaviour}_{metric}_heatmap.png"
        plt.savefig(save_path, dpi=300)
        plt.savefig(save_path.path.with_suffix(".pdf"), dpi=300)
    plt.show()