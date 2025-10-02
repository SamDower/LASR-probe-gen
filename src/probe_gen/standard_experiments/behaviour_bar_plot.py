import matplotlib.pyplot as plt
import numpy as np
import textwrap

from probe_gen.config import ConfigDict
from probe_gen.probes.wandb_interface import load_probe_eval_dict_batch
from probe_gen.standard_experiments.hyperparameter_search import (
    get_best_hyperparams_for_train_setup
)


def get_bar_chart_results_table_from_wandb(train_setup, train_gen_methods, test_gen_method, train_OOD):
    """
    Gets the results table from wandb for the probes specified in the train_setup.
    Args:
        train_setup (list): 
            [probe_type, behaviour, [ID datasource, OOD datasource], activations_model, off_policy_model]
        ...
    """
    tr = train_setup
    
    lookup_dicts = []
    for i in range(len(tr)):
        for train_gen_method in train_gen_methods:
            # Get the best hyperparameters for each probe if not provided
            response_model = tr[i][4] if train_gen_method == "off_policy" else tr[i][3]
            # format: [probe_type, behaviour, datasource, activations_model, generation_method, response_model, mode]
            full_tr_i = [tr[i][0], tr[i][1], tr[i][2][0], tr[i][3], train_gen_method, response_model, "train"]
            cfg = get_best_hyperparams_for_train_setup([full_tr_i])[0][-1]

            search_dict = {
                "config.probe/type": tr[i][0],
                "behaviour": tr[i][1],
                "train/datasource": tr[i][2][1] if train_OOD else tr[i][2][0],
                "train/generation_method": train_gen_method,
                "train/response_model": response_model,
                "test/datasource": tr[i][2][0],
                "test/generation_method": test_gen_method,
                "test/response_model": response_model,
                "config.layer": cfg.layer,
                "config.probe/use_bias": cfg.use_bias,
                "config.probe/normalize": cfg.normalize,
                "config.activations_model": full_tr_i[3],
                "state": "finished"  # Only completed runs
            }
            if full_tr_i[0] == "mean":
                search_dict["config.probe/C"] = cfg.C
            elif "torch" in full_tr_i[0]:
                search_dict["config.probe/lr"] = cfg.lr
                search_dict["config.probe/weight_decay"] = cfg.weight_decay
                
            lookup_dicts.append(search_dict)

    results_list = load_probe_eval_dict_batch(lookup_dicts)
    results_table = np.array([results_list[i]['roc_auc'] for i in range(len(results_list))]).reshape(len(tr), len(train_gen_methods)).transpose()
    return results_table


def plot_behaviour_barchart(
    train_setup, 
    train_OOD=False, 
    test_incentivised=False, 
    add_mean_summary=False, 
    title=None,
    xlabel="Behaviour",
    ylabel="Test AUROC",
    save_path = None,
    figsize=(12, 3), 
    dpi=300,
    legend_loc="upper right",
    extra_whitespace=1
    ):
    """
    Plots a bar chart of the results of the probes specified in the train_setup list.
    Args:
        train_setup (list): 
            [probe_type, behaviour, [ID datasource, OOD datasource], activations_model, off_policy_model]
        ...
    """
    small_gap = 0.2
    big_gap = 0.5

    # Get all results by querying wandb for all run configs
    if test_incentivised:
        group_labels =  ['On-Policy Incentivised', 'On-Policy Prompted', 'Off-Policy']
        train_gen_methods = ['incentivised', 'prompted', 'off_policy']
        test_gen_method = 'incentivised'
    else:
        group_labels =  ['On-Policy', 'On-Policy Incentivised', 'On-Policy Prompted', 'Off-Policy']
        train_gen_methods = ['on_policy', 'incentivised', 'prompted', 'off_policy']
        test_gen_method = 'on_policy'
    print("Fetching results...")
    results_table = get_bar_chart_results_table_from_wandb(train_setup, train_gen_methods, test_gen_method, train_OOD)
    print("Fetched.")
    
    x = np.arange(len(train_setup))  # Positions 0, 1, 2, ..., 8
    masked_array = np.ma.masked_equal(results_table, 0)
    row_means = np.ma.mean(masked_array, axis=1)
    row_stds = np.ma.std(masked_array, axis=1)

    behaviour_labels = [f"{train_setup[i][1]}({train_setup[i][2][1] if train_OOD else train_setup[i][2][0]})" 
                        for i in range(len(train_setup))]
    if add_mean_summary:
        behaviour_labels.append('Mean ± MSE')

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    train_colors = ['#264653', '#2A9D8F', '#E76F51', '#F18F01']  if not test_incentivised else ['#2A9D8F', '#E76F51', '#F18F01']

    # Create the grouped bars - separate first groups from mean group
    pattern = "xx" if train_OOD else ""
    num_groups = results_table.shape[0]
    bar_width = (1 - small_gap) / num_groups
    for i in range(num_groups):
        group_offset = (i - num_groups / 2 + 0.5) * bar_width

        ax.bar(x + group_offset, results_table[i], bar_width, label=group_labels[i], color=train_colors[i % 4], alpha=0.8, hatch=pattern)
        
        if add_mean_summary:
            ax.bar(np.array([len(train_setup) + big_gap - small_gap]) + group_offset, row_means[i], bar_width, color=train_colors[i % 4], alpha=0.8, 
                     yerr=row_stds[i], capsize=5, error_kw={'elinewidth': 2, 'capthick': 2}, hatch=pattern)

    # Customize the plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is None:
        title = f"Linear Probes Trained on {'OOD' if train_OOD else 'ID'} Training Sets, Evaluated on  On-Policy {'Incentivised ' if test_incentivised else ''}Test Sets for Each Behaviour"
    ax.set_title(title)
    
    x_ticks = np.concatenate([x, [len(train_setup) + big_gap - small_gap]]) if add_mean_summary else x
    ax.set_xticks(x_ticks)
    wrapped_labels = ['\n'.join(textwrap.wrap(label, width=14)) for label in behaviour_labels]
    ax.set_xticklabels(wrapped_labels)
    ax.tick_params(axis='x', labelsize=8)
    
    current_xlim = ax.get_xlim()
    ax.set_xlim(current_xlim[0], current_xlim[1] + extra_whitespace)
    
    #ax.legend(loc='upper right', title="ID Training Set")
    ax.legend(loc=legend_loc, title=f"{'OOD' if train_OOD else 'ID'} Training Set", fontsize=9, title_fontsize=9)

    # Add a grid for better readability
    ax.grid(True, alpha=0.3, axis='y')

    # Adjust layout and display
    plt.ylim(0.5, 1)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)
    plt.show()


def plot_mean_summary_barchart(
    train_setup, 
    test_incentivised=False, 
    title=None,
    xlabel=None,
    ylabel="Test AUROC",
    save_path = None,
    figsize=(12, 6), 
    dpi=300,
    legend_loc="upper right",
    extra_whitespace=1
    ):
    """
    Plots a summary mean bar chart of the results of the probes specified in the train_setup list.
    Args:
        train_setup (list): 
            [probe_type, behaviour, [ID datasource, OOD datasource], activations_model, off_policy_model]
        ...
    """
    small_gap = 0.2

    # Get all results by querying wandb for all run configs
    if test_incentivised:
        group_labels =  ['On-Policy Incentivised', 'On-Policy Prompted', 'Off-Policy']
        train_gen_methods = ['incentivised', 'prompted', 'off_policy']
        test_gen_method = 'incentivised'
    else:
        group_labels =  ['On-Policy', 'On-Policy Incentivised', 'On-Policy Prompted', 'Off-Policy']
        train_gen_methods = ['on_policy', 'incentivised', 'prompted', 'off_policy']
        test_gen_method = 'on_policy'
    means = np.full((len(train_gen_methods), 2), 0.6, dtype=float)
    standard_errors = np.full((len(train_gen_methods), 2), 0.6, dtype=float)
    for i, train_OOD in enumerate([False, True]):
        print("Fetching results...")
        results_table = get_bar_chart_results_table_from_wandb(train_setup, train_gen_methods, test_gen_method, train_OOD)
        masked_array = np.ma.masked_equal(results_table, 0)
        means[:,i] = np.ma.mean(masked_array, axis=1)
        standard_errors[:,i] = np.ma.std(masked_array, axis=1) / np.sqrt(masked_array.shape[1])
    print("Fetched")
    x = np.arange(2)  # Positions 0, 1, 2, ..., 8

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    train_colors = ['#264653', '#2A9D8F', '#E76F51', '#F18F01']  if not test_incentivised else ['#2A9D8F', '#E76F51', '#F18F01']

    # Create the grouped bars - separate first groups from mean group
    num_groups = results_table.shape[0]
    bar_width = (1 - small_gap) / num_groups
    for i in range(num_groups):
        group_offset = (i - num_groups / 2 + 0.5) * bar_width
        bars = ax.bar(x + group_offset, means[i], bar_width, label=group_labels[i], color=train_colors[i % 4], alpha=0.8, 
                    yerr=standard_errors[i], capsize=4) #error_kw={'elinewidth': 2, 'capthick': 2}, 
        for j in [1]:
            bars[j].set_hatch('xx')

    # Customize the plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is None:
        if not test_incentivised:
            title = "Mean (and standard error) of Linear Probes, Evaluated on ID On-Policy Sets Across All Behaviours" 
        else:
            title = "Mean (and standard error) of Linear Probes, Evaluated on ID On-Policy Incentivised Sets Across All Behaviours" 
    ax.set_title(title)
    ax.set_xticks(x)

    labels = ["Trained on ID Data", "Trained on OOD Data"]
    wrapped_labels = ['\n'.join(textwrap.wrap(label, width=14)) for label in labels]
    ax.set_xticklabels(wrapped_labels)

    current_xlim = ax.get_xlim()
    ax.set_xlim(current_xlim[0], current_xlim[1] + extra_whitespace)

    #ax.legend(loc='upper right', title="ID Training Set")
    ax.legend(loc=legend_loc, title="Training Set", fontsize=9, title_fontsize=9)

    # Add a grid for better readability
    ax.grid(True, alpha=0.3, axis='y')

    # Adjust layout and display
    plt.ylim(0.5, 1)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)
    plt.show()
