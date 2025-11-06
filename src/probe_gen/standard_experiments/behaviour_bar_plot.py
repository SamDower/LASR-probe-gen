import sys
import textwrap
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from tqdm import tqdm

from probe_gen.probes.wandb_interface import (
    load_probe_eval_dicts_as_df,  #,load_probe_eval_dict_batch 
)
from probe_gen.standard_experiments.hyperparameter_search import (
    get_best_hyperparams_for_train_setup,
)

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def get_bar_chart_results_table_from_wandb(train_setup, train_gen_methods, test_gen_method, train_OOD):
    """
    Gets the results table from wandb for the probes specified in the train_setup.
    Args:
        train_setup (list): 
            [probe_type, behaviour, [ID datasource, OOD datasource], activations_model, [ID off_policy_model, OOD off_policy_model]]
        ...
    """
    tr = train_setup
    
    results_list = []
    for i in tqdm(range(len(tr))):
        for train_gen_method in train_gen_methods:
            # Get the best hyperparameters for each probe if not provided
            train_datasource = tr[i][2][1] if train_OOD else tr[i][2][0]
            off_policy_model = tr[i][4][1] if train_OOD else tr[i][4][0]
            response_model = off_policy_model if train_gen_method == "off_policy" else tr[i][3]
            # format: [probe_type, behaviour, datasource, activations_model, generation_method, response_model, mode]
            full_tr_i = [tr[i][0], tr[i][1], tr[i][2][0], tr[i][3], train_gen_method, response_model, "train"]
            cfg = get_best_hyperparams_for_train_setup([full_tr_i])[0][-1]

            search_dict = {
                "config.probe/type": tr[i][0],
                "config.behaviour": tr[i][1],
                "config.train/datasource": train_datasource,
                "config.train/generation_method": train_gen_method,
                "config.train/response_model": response_model,
                "config.test/datasource": tr[i][2][0],
                "config.test/generation_method": test_gen_method,
                "config.test/response_model": tr[i][3],
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
                
            print(search_dict)

            run_df = load_probe_eval_dicts_as_df(search_dict)
            results_list.append(run_df['metric_roc_auc'].iloc[-1])

    results_table = np.array(results_list).reshape(len(tr), len(train_gen_methods)).transpose()
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
    extra_whitespace=1,
    probe_type="mean",
    do_seperator_line=True, 
    ):
    """
    Plots a bar chart of the results of the probes specified in the train_setup list.
    Args:
        train_setup (list): 
            [probe_type, behaviour, [ID datasource, OOD datasource], activations_model, [ID off_policy_model, OOD off_policy_model]]
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
        group_labels =  ['On-Policy Natural', 'On-Policy Incentivised', 'On-Policy Prompted', 'Off-Policy']
        train_gen_methods = ['on_policy', 'incentivised', 'prompted', 'off_policy']
        test_gen_method = 'on_policy'
    print("Fetching results...")
    results_table = get_bar_chart_results_table_from_wandb(train_setup, train_gen_methods, test_gen_method, train_OOD)
    print("Fetched.")
    
    x = np.arange(len(train_setup))  # Positions 0, 1, 2, ..., 8
    masked_array = np.ma.masked_equal(results_table, 0)
    row_means = np.ma.mean(masked_array, axis=1)
    row_stds = np.ma.std(masked_array, axis=1)

    behaviour_labels = [f"{train_setup[i][1].capitalize()} ({train_setup[i][2][1].capitalize() if train_OOD else train_setup[i][2][0].capitalize()})" 
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
        title = f"{'Linear' if probe_type == 'mean' else 'Attention'} Probes With {'Different-Domain' if train_OOD else 'Same-Domain'} Train Set, Evaluated Against On-Policy {'Incentivised ' if test_incentivised else 'Natural '}"
    ax.set_title(title)
    
    x_ticks = np.concatenate([x, [len(train_setup) + big_gap - small_gap]]) if add_mean_summary else x
    ax.set_xticks(x_ticks)
    wrapped_labels = ['\n'.join(textwrap.wrap(label, width=14)) for label in behaviour_labels]
    ax.set_xticklabels(wrapped_labels)
    ax.tick_params(axis='x', labelsize=9)
    
    current_xlim = ax.get_xlim()
    ax.set_xlim(current_xlim[0], current_xlim[1] + extra_whitespace)
    
    #ax.legend(loc='upper right', title="ID Training Set")
    ax.legend(loc=legend_loc, fontsize=10, title_fontsize=11)

    # Add a grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.title.set_fontsize(16)     # change font size separately
    ax.xaxis.label.set_size(13)   # x-axis label font size
    ax.yaxis.label.set_size(13)   # y-axis label font size
    
    if test_incentivised and (do_seperator_line != False):
        # Insert vertical dashed red line
        split_index = len(train_setup) - 4  # index where the last 4 behaviours begin
        ax.axvline(
            x=split_index - 0.5,   # -0.5 so it appears between bars
            color='red',
            linestyle='--',
            linewidth=1.5,
            alpha=0.8,
        )

    # Adjust layout and display
    plt.ylim(0.5, 1)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)
    plt.show()


def get_bar_chart_results_table_from_wandb_same_test_train(train_setup):
    """
    Gets the results table from wandb for the probes specified in the train_setup.
    Args:
        train_setup (list): 
            [probe_type, behaviour, [ID datasource, OOD datasource], activations_model, [ID off_policy_model, OOD off_policy_model]]
        ...
    """
    tr = train_setup
    
    train_gen_methods = ['on_policy', 'incentivised', 'prompted', 'off_policy']
    results_list = []
    for i in tqdm(range(len(tr))):
        for train_gen_method in train_gen_methods:
            if tr[i][1] in ["deception", "sandbagging"] and train_gen_method == "on_policy":
                results_list.append(0)
                continue
            
            # Get the best hyperparameters for each probe if not provided
            train_datasource = tr[i][2][0]
            off_policy_model = tr[i][4][0]
            response_model = off_policy_model if train_gen_method == "off_policy" else tr[i][3]
            # format: [probe_type, behaviour, datasource, activations_model, generation_method, response_model, mode]
            full_tr_i = [tr[i][0], tr[i][1], tr[i][2][0], tr[i][3], train_gen_method, response_model, "train"]
            cfg = get_best_hyperparams_for_train_setup([full_tr_i])[0][-1]

            search_dict = {
                "config.probe/type": tr[i][0],
                "config.behaviour": tr[i][1],
                "config.train/datasource": train_datasource,
                "config.train/generation_method": train_gen_method,
                "config.train/response_model": response_model,
                "config.test/datasource": train_datasource,
                "config.test/generation_method": train_gen_method,
                "config.test/response_model": response_model,
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
                
            print(search_dict)

            run_df = load_probe_eval_dicts_as_df(search_dict)
            results_list.append(run_df['metric_roc_auc'].iloc[-1])

    results_table = np.array(results_list).reshape(len(tr), len(train_gen_methods)).transpose()
    return results_table


def plot_behaviour_barchart_same_test_train(
    train_setup, 
    train_OOD=False, 
    add_mean_summary=False, 
    title=None,
    xlabel="Behaviour",
    ylabel="Test AUROC",
    save_path = None,
    figsize=(12, 3), 
    dpi=300,
    legend_loc="upper right",
    extra_whitespace=1,
    probe_type="mean",
    do_seperator_line=False, 
    ):
    """
    Plots a bar chart of the results of the probes specified in the train_setup list.
    Args:
        train_setup (list): 
            [probe_type, behaviour, [ID datasource, OOD datasource], activations_model, [ID off_policy_model, OOD off_policy_model]]
        ...
    """
    small_gap = 0.2
    big_gap = 0.5

    # Get all results by querying wandb for all run configs
    group_labels =  ['On-Policy Natural', 'On-Policy Incentivised', 'On-Policy Prompted', 'Off-Policy']
    print("Fetching results...")
    results_table = get_bar_chart_results_table_from_wandb_same_test_train(train_setup)
    print("Fetched.")
    
    x = np.arange(len(train_setup))  # Positions 0, 1, 2, ..., 8
    masked_array = np.ma.masked_equal(results_table, 0)
    row_means = np.ma.mean(masked_array, axis=1)
    row_stds = np.ma.std(masked_array, axis=1)

    behaviour_labels = [f"{train_setup[i][1].capitalize()} ({train_setup[i][2][1].capitalize() if train_OOD else train_setup[i][2][0].capitalize()})" 
                        for i in range(len(train_setup))]
    if add_mean_summary:
        behaviour_labels.append('Mean ± MSE')

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    train_colors = ['#3B5BA5', '#6C8CD5', '#8E7DBE', '#C6B4CE']

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
        title = f"{'Linear' if probe_type == 'mean' else 'Attention'} Probes With Same Train and Test Sets"
    ax.set_title(title)
    
    x_ticks = np.concatenate([x, [len(train_setup) + big_gap - small_gap]]) if add_mean_summary else x
    ax.set_xticks(x_ticks)
    wrapped_labels = ['\n'.join(textwrap.wrap(label, width=14)) for label in behaviour_labels]
    ax.set_xticklabels(wrapped_labels)
    ax.tick_params(axis='x', labelsize=9)
    
    current_xlim = ax.get_xlim()
    ax.set_xlim(current_xlim[0], current_xlim[1] + extra_whitespace)
    
    #ax.legend(loc='upper right', title="ID Training Set")
    ax.legend(loc=legend_loc, fontsize=10, title_fontsize=11)

    # Add a grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.title.set_fontsize(16)     # change font size separately
    ax.xaxis.label.set_size(13)   # x-axis label font size
    ax.yaxis.label.set_size(13)   # y-axis label font size
    
    if do_seperator_line:
        # Insert vertical dashed red line
        split_index = len(train_setup) - 4  # index where the last 4 behaviours begin
        ax.axvline(
            x=split_index - 0.5,   # -0.5 so it appears between bars
            color='red',
            linestyle='--',
            linewidth=1.5,
            alpha=0.8,
        )

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
    ylabel="Mean Test AUROC (± SEM)",
    save_path = None,
    figsize=(5, 3), 
    dpi=300,
    legend_loc="upper right",
    extra_whitespace=1,
    probe_type="mean",
    verbose=False
    ):
    """
    Plots a summary mean bar chart of the results of the probes specified in the train_setup list.
    Args:
        train_setup (list): 
            [probe_type, behaviour, [ID datasource, OOD datasource], activations_model, [ID off_policy_model, OOD off_policy_model]]
        ...
    """
    small_gap = 0.2

    # Get all results by querying wandb for all run configs
    if test_incentivised:
        group_labels =  ['On-Policy Incentivised', 'On-Policy Prompted', 'Off-Policy']
        train_gen_methods = ['incentivised', 'prompted', 'off_policy']
        test_gen_method = 'incentivised'
    else:
        group_labels =  ['On-Policy Natural', 'On-Policy Incentivised', 'On-Policy Prompted', 'Off-Policy']
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
    
    if verbose:
        print(f"Means: {means}")
        print(f"Standard errors: {standard_errors}")

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
        title = f"{'Linear' if probe_type == 'mean' else 'Attention'} Probes Evaluated Against On-Policy {'Incentivised ' if test_incentivised else 'Natural '}"
    ax.set_title(title)
    ax.set_xticks(x)

    labels = ["Same-Domain Train Set", "Different-Domain Train Set"]
    wrapped_labels = ['\n'.join(textwrap.wrap(label, width=14)) for label in labels]
    ax.set_xticklabels(wrapped_labels)

    current_xlim = ax.get_xlim()
    ax.set_xlim(current_xlim[0], current_xlim[1] + extra_whitespace)

    #ax.legend(loc='upper right', title="ID Training Set")
    ax.legend(loc=legend_loc, fontsize=9, title_fontsize=11)

    # Add a grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.title.set_fontsize(13)     # change font size separately
    ax.xaxis.label.set_size(12)   # x-axis label font size
    ax.yaxis.label.set_size(12)   # y-axis label font 
    
    # Adjust layout and display
    plt.ylim(0.5, 1.19)
    plt.yticks(np.linspace(0.5, 1, 6))

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)
    plt.show()


def plot_mean_summary_dotchart(
    train_setup, 
    test_incentivised=False, 
    title=None,
    xlabel=None,
    ylabel="Test AUROC",
    save_path = None,
    figsize=(7, 4), 
    dpi=300,
    probe_type="mean",
    draw_blobs=True
    ):
    if test_incentivised:
        group_labels =  ['On-Policy Incentivised', 'On-Policy Prompted', 'Off-Policy']
        train_gen_methods = ['incentivised', 'prompted', 'off_policy']
        test_gen_method = 'incentivised'
    else:
        group_labels =  ['On-Policy Natural', 'On-Policy Incentivised', 'On-Policy Prompted', 'Off-Policy']
        train_gen_methods = ['on_policy', 'incentivised', 'prompted', 'off_policy']
        test_gen_method = 'on_policy'

    train_colors = ['#264653', '#2A9D8F', '#E76F51', '#F18F01'] if not test_incentivised else ['#2A9D8F', '#E76F51', '#F18F01']

    # Fetch data
    print("Fetching results...")
    results_same = get_bar_chart_results_table_from_wandb(train_setup, train_gen_methods, test_gen_method, train_OOD=False)
    results_diff = get_bar_chart_results_table_from_wandb(train_setup, train_gen_methods, test_gen_method, train_OOD=True)
    print("Fetched")

    # Ensure arrays are numpy arrays of float and use np.nan for missing if needed
    results_same = np.array(results_same, dtype=float)
    results_diff = np.array(results_diff, dtype=float)

    # If your missing values are represented by 0 (as before), convert them to np.nan so plotting skips them:
    results_same[results_same == 0] = np.nan
    results_diff[results_diff == 0] = np.nan

    title=None
    xlabel=None
    ylabel="Test AUROC"
    figsize=(7, 4)
    dpi=300
    save_path="linear_id_vs_ood_onpolicy.pdf"
    jitter=0.08
    alpha_dots=0.75
    alpha_lines=0.35
    blob_alpha=0.15  # transparency for background blobs
    blob_scale=0.25  # how wide the blobs appear

    num_policies, n_cases = results_same.shape
    assert results_diff.shape == (num_policies, n_cases)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Pre-generate jitter offsets (same jittering logic across domains)
    jitter_same = np.random.normal(0, jitter, size=(num_policies, n_cases))
    jitter_diff = np.random.normal(0, jitter, size=(num_policies, n_cases))

    # X positions
    x_positions_same = np.arange(num_policies)
    x_positions_diff = x_positions_same + num_policies + 1  # +1 gap between groups

    def draw_blob(ax, center_x, y_vals, color, scale=blob_scale, alpha=blob_alpha):
        y_vals = y_vals[(y_vals != 0) & ~np.isnan(y_vals)]
        if len(y_vals) < 3:
            return
        kde = gaussian_kde(y_vals)
        y_grid = np.linspace(min(y_vals), max(y_vals), 100)
        dens = kde(y_grid)
        dens = dens / dens.max() * scale  # normalize width
        ax.fill_betweenx(
            y_grid,
            center_x - dens,
            center_x + dens,
            color=color,
            alpha=alpha,
            linewidth=0,       # ✅ no stroke width
            edgecolor='none',  # ✅ suppress any outline
            zorder=0
        )

    if draw_blobs:
        # --- DRAW BLOBS (before dots/lines) ---
        for i in range(num_policies):
            color = train_colors[i % len(train_colors)]
            draw_blob(ax, x_positions_same[i], results_same[i, :], color)
            draw_blob(ax, x_positions_diff[i], results_diff[i, :], color)

    # --- CONTINUOUS CONNECTED LINES ACROSS DOMAINS ---
    for j in range(n_cases):
        y_same = results_same[:, j]
        y_diff = results_diff[:, j]
        y_combined = np.concatenate([y_same, y_diff])
        valid = (y_combined != 0) & ~np.isnan(y_combined)
        if np.any(valid):
            xs_same = x_positions_same + jitter_same[:, j]
            xs_diff = x_positions_diff + jitter_diff[:, j]
            xs_combined = np.concatenate([xs_same, xs_diff])
            ax.plot(xs_combined[valid], y_combined[valid],
                    color="gray", alpha=alpha_lines, linewidth=0.9, zorder=1)

    # --- DOTS (Same-domain) ---
    for i in range(num_policies):
        color = train_colors[i % len(train_colors)]
        y_vals = results_same[i, :]
        valid = (y_vals != 0) & ~np.isnan(y_vals)
        xs = x_positions_same[i] + jitter_same[i, valid]
        ax.scatter(xs, y_vals[valid], color=color, alpha=alpha_dots, s=24, edgecolors='none', zorder=2)

    # --- DOTS (Diff-domain) ---
    for i in range(num_policies):
        color = train_colors[i % len(train_colors)]
        y_vals = results_diff[i, :]
        valid = (y_vals != 0) & ~np.isnan(y_vals)
        xs = x_positions_diff[i] + jitter_diff[i, valid]
        ax.scatter(xs, y_vals[valid], color=color, alpha=alpha_dots, s=24, edgecolors='none', zorder=2)

    # --- Formatting ---
    ax.axvline(x=num_policies, color="gray", alpha=0.2, linewidth=1)

    patches = [mpatches.Patch(color=train_colors[i], label=group_labels[i]) for i in range(num_policies)]
    ax.legend(handles=patches, loc="upper right", fontsize=10, title_fontsize=12)

    ax.set_xticks(list(x_positions_same) + list(x_positions_diff))
    ax.set_xticklabels([""] * (len(x_positions_same) + len(x_positions_diff)))

    # Add centered domain labels
    mid_same = np.mean(x_positions_same)
    mid_diff = np.mean(x_positions_diff)
    ax.text(mid_same, ax.get_ylim()[0] - 0.03, "Same-Domain Train Set", ha="center", va="top", fontsize=12)
    ax.text(mid_diff, ax.get_ylim()[0] - 0.03, "Different-Domain Train Set", ha="center", va="top", fontsize=12)

    title = f"{'Linear' if probe_type == 'mean' else 'Attention'} Probes Evaluated Against On-Policy {'Incentivised ' if test_incentivised else 'Natural '}"
    ax.set_title(title)

    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True, axis='y', alpha=0.25)
    
        
    # Adjust layout and display
    plt.ylim(0.5, 1.19)
    plt.yticks(np.linspace(0.5, 1, 6))

    ax.title.set_fontsize(15)
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)
    plt.show()


def plot_mean_summary_barchart_for_including_prompts_or_not(
    train_setup, 
    title=None,
    xlabel=None,
    ylabel="Mean Test AUROC (± SEM)",
    save_path = None,
    figsize=(8, 3), 
    dpi=300,
    legend_loc="upper right",
    extra_whitespace=1
    ):
    """
    Plots a summary mean bar chart of the results of the probes specified in the train_setup list.
    Args:
        train_setup (list): 
            [probe_type, behaviour, [ID datasource, OOD datasource], activations_model, [ID off_policy_model, OOD off_policy_model]]
        ...
    """
    small_gap = 0.2

    # Get all results by querying wandb for all run configs
    group_labels =  ['On-Policy Incentivised (incentive not included)', 'On-Policy Incentivised (incentive included)', 'On-Policy Prompted (prompt not included)', 'On-Policy Prompted (prompt included)']
    train_gen_methods = ['incentivised', 'incentivised_included', 'prompted', 'prompted_included']
    test_gen_method = 'on_policy'
    means = np.full((len(train_gen_methods), 1), 0.6, dtype=float)
    standard_errors = np.full((len(train_gen_methods), 1), 0.6, dtype=float)
    print("Fetching results...")
    results_table = get_bar_chart_results_table_from_wandb(train_setup, train_gen_methods, test_gen_method, False)
    masked_array = np.ma.masked_equal(results_table, 0)
    means[:,0] = np.ma.mean(masked_array, axis=1)
    standard_errors[:,0] = np.ma.std(masked_array, axis=1) / np.sqrt(masked_array.shape[1])
    print("Fetched")
    x = np.arange(1)  # Positions 0, 1, 2, ..., 8

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    train_colors = ['#2A9D8F', '#8DC7BF', '#E76F51', '#F2B5A5']

    # Create the grouped bars - separate first groups from mean group
    num_groups = results_table.shape[0]
    bar_width = (1 - small_gap) / num_groups
    for i in range(num_groups):
        group_offset = (i - num_groups / 2 + 0.5) * bar_width
        bars = ax.bar(x + group_offset, means[i], bar_width, label=group_labels[i], color=train_colors[i % 4], alpha=0.8, 
                    yerr=standard_errors[i], capsize=4) #error_kw={'elinewidth': 2, 'capthick': 2}, 

    # Customize the plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is None:
        title = "Effect of including prompt / incentive activations on linear probe generalization"
    ax.set_title(title)
    ax.set_xticks(x)

    labels = ["Training Set"]
    wrapped_labels = ['\n'.join(textwrap.wrap(label, width=14)) for label in labels]
    ax.set_xticklabels(wrapped_labels)

    current_xlim = ax.get_xlim()
    ax.set_xlim(current_xlim[0], current_xlim[1] + extra_whitespace)

    #ax.legend(loc='upper right', title="ID Training Set")
    ax.legend(loc=legend_loc, title="Training Set", fontsize=9, title_fontsize=11)

    # Add a grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.title.set_fontsize(13)     # change font size separately
    ax.xaxis.label.set_size(12)   # x-axis label font size
    ax.yaxis.label.set_size(12)   # y-axis label font size

    # Adjust layout and display
    plt.ylim(0.5, 1)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)
    plt.show()


def get_combined_bar_chart_results_table_from_wandb(train_setup, train_gen_methods, test_gen_method):
    """
    Gets the results table from wandb for combined probe experiments.
    Args:
        train_setup (list): 
            [probe_type, behaviour, [datasource1, datasource2, ...], activations_model, test_datasource]
            Example: ["mean", "lists", ["writingprompts", "ultrachat"], "llama_3b", "shakespeare"]
        train_gen_methods (list): List of generation methods to query, e.g., ['on_policy', 'incentivised']
        test_gen_method (str): Generation method for test set, e.g., 'on_policy'
    """
    tr = train_setup
    
    results_list = []
    for i in tqdm(range(len(tr)), desc="Querying WandB for combined probes"):
        probe_type = tr[i][0]
        behaviour = tr[i][1]
        datasource_pair = tr[i][2]
        activations_model = tr[i][3]
        test_datasource = tr[i][4]
        response_model = activations_model
        
        # Get hyperparameters from first datasource in the pair
        # format: [probe_type, behaviour, datasource, activations_model, generation_method, response_model, mode]
        # We use the first datasource to get best hyperparameters (they should be the same)
        full_tr_i = [probe_type, behaviour, datasource_pair[0], activations_model, 
                     train_gen_methods[0], response_model, "train"]
        cfg = get_best_hyperparams_for_train_setup([full_tr_i])[0][-1]
        
        if len(datasource_pair) > 1:
            for train_gen_method in train_gen_methods:
                # Create combined train datasource name
                combined_train_desc = "+".join([f"{ds}_{train_gen_method}" for ds in datasource_pair])
                
                search_dict = {
                    "config.probe/type": probe_type,
                    "config.behaviour": behaviour,
                    "config.train/datasource": combined_train_desc,
                    "config.train/generation_method": "combined",
                    "config.train/response_model": response_model,
                    "config.test/datasource": test_datasource,
                    "config.test/generation_method": test_gen_method,
                    "config.test/response_model": response_model,
                    "config.layer": cfg.layer,
                    "config.probe/use_bias": cfg.use_bias,
                    "config.probe/normalize": cfg.normalize,
                    "config.activations_model": activations_model,
                    "state": "finished"
                }
                
                if probe_type == "mean":
                    search_dict["config.probe/C"] = cfg.C
                elif "torch" in probe_type:
                    search_dict["config.probe/lr"] = cfg.lr
                    search_dict["config.probe/weight_decay"] = cfg.weight_decay
                
                run_df = load_probe_eval_dicts_as_df(search_dict)
                
                if len(run_df) > 0:
                    results_list.append(run_df['metric_roc_auc'].iloc[-1])
                else:
                    print(f"⚠ No results found for {behaviour} with {train_gen_method}")
                    results_list.append(np.nan)
        else:
            for train_gen_method in train_gen_methods:
                # Create combined train datasource name
                combined_train_desc = "+".join([f"{ds}_{train_gen_method}" for ds in datasource_pair])
                
                search_dict = {
                    "config.probe/type": probe_type,
                    "config.behaviour": behaviour,
                    "config.train/datasource": datasource_pair[0],
                    "config.train/generation_method": train_gen_method,
                    "config.test/datasource": test_datasource,
                    "config.test/generation_method": test_gen_method,
                    "config.test/response_model": response_model,
                    "config.layer": cfg.layer,
                    "config.probe/use_bias": cfg.use_bias,
                    "config.probe/normalize": cfg.normalize,
                    "config.activations_model": activations_model,
                    "state": "finished"
                }
                
                if probe_type == "mean":
                    search_dict["config.probe/C"] = cfg.C
                elif "torch" in probe_type:
                    search_dict["config.probe/lr"] = cfg.lr
                    search_dict["config.probe/weight_decay"] = cfg.weight_decay
                
                run_df = load_probe_eval_dicts_as_df(search_dict)
                
                if len(run_df) > 0:
                    results_list.append(run_df['metric_roc_auc'].iloc[-1])
                else:
                    print(f"⚠ No results found for {behaviour} with {train_gen_method}")
                    results_list.append(np.nan)
    
    results_table = np.array(results_list).reshape(len(tr), len(train_gen_methods)).transpose()
    return results_table


def plot_combined_behaviour_barchart(
    train_setup,
    test_incentivised=False,
    add_mean_summary=False,
    title=None,
    xlabel="Combined Training Sets",
    ylabel="Test AUROC",
    save_path=None,
    figsize=(12, 3),
    dpi=300,
    legend_loc="upper right",
    extra_whitespace=1,
    probe_type="mean",
):
    """
    Plots a bar chart for combined probe experiments (training on multiple datasets).
    
    Args:
        train_setup (list of lists): Each element is:
            [probe_type, behaviour, [datasource1, datasource2, ...], activations_model, test_datasource]
            Example: [["mean", "lists", ["writingprompts", "ultrachat"], "llama_3b", "shakespeare"]]
        test_incentivised (bool): If True, test on incentivised data; otherwise on_policy
        add_mean_summary (bool): If True, add a summary bar showing mean and std
        title (str): Plot title (auto-generated if None)
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        save_path (str): Path to save figure (if None, don't save)
        figsize (tuple): Figure size
        dpi (int): DPI for saved figure
        legend_loc (str): Legend location
        extra_whitespace (float): Extra whitespace on right side
        probe_type (str): "mean" or "attention_torch"
    """
    small_gap = 0.2
    big_gap = 0.5

    # Get all results by querying wandb for all run configs
    group_labels = ['On-Policy Natural', 'On-Policy Incentivised', 'On-Policy Prompted', 'Off-Policy']
    train_gen_methods = ['on_policy', 'incentivised', 'prompted', 'off_policy']
    test_gen_method = 'on_policy'

    # train_setup = [
    #     [probe_type, behaviour, [datasources[0], datasources[1], datasources[2]], activations_model, datasources[0]],
    #     [probe_type, behaviour, [datasources[0], datasources[1]], activations_model, datasources[0]],
    #     [probe_type, behaviour, [datasources[1], datasources[2]], activations_model, datasources[0]],
    #     [probe_type, behaviour, [datasources[0]], activations_model, datasources[0]],
    #     [probe_type, behaviour, [datasources[1]], activations_model, datasources[0]],
    #     [probe_type, behaviour, [datasources[2]], activations_model, datasources[0]],
    # ]
    
    print("Fetching combined probe results from WandB...")
    results_table = get_combined_bar_chart_results_table_from_wandb(
        train_setup, train_gen_methods, test_gen_method
    )
    print("Fetched.")
    
    x = np.arange(len(train_setup))
    masked_array = np.ma.masked_invalid(results_table)
    row_means = np.ma.mean(masked_array, axis=1)
    row_stds = np.ma.std(masked_array, axis=1)

    # Create labels for each combined training set
    behaviour_labels = []
    for i in range(len(train_setup)):
        behaviour = train_setup[i][1].capitalize()
        datasources = train_setup[i][2]
        test_ds = train_setup[i][4].capitalize()
        
        # Abbreviate datasource names
        ds_abbrev = "+".join([ds[:3].upper() for ds in datasources])
        behaviour_labels.append(f"{behaviour}\n({ds_abbrev}→{test_ds[:3].upper()})")
    
    if add_mean_summary:
        behaviour_labels.append('Mean ± Std')

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    train_colors = ['#264653', '#2A9D8F', '#E76F51', '#F18F01'] if not test_incentivised else ['#2A9D8F', '#E76F51', '#F18F01']

    # Create the grouped bars
    num_groups = results_table.shape[0]
    bar_width = (1 - small_gap) / num_groups
    
    for i in range(num_groups):
        group_offset = (i - num_groups / 2 + 0.5) * bar_width
        ax.bar(x + group_offset, results_table[i], bar_width, 
               label=group_labels[i], color=train_colors[i % 4], alpha=0.8)
        
        if add_mean_summary:
            ax.bar(np.array([len(train_setup) + big_gap - small_gap]) + group_offset, 
                   row_means[i], bar_width, color=train_colors[i % 4], alpha=0.8,
                   yerr=row_stds[i], capsize=5, 
                   error_kw={'elinewidth': 2, 'capthick': 2})

    # Customize the plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is None:
        title = f"{'Linear' if probe_type == 'mean' else 'Attention'} Probes on Combined Training Sets, Evaluated Against {'Incentivised' if test_incentivised else 'On-Policy Natural'}"
    ax.set_title(title)
    
    x_ticks = np.concatenate([x, [len(train_setup) + big_gap - small_gap]]) if add_mean_summary else x
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(behaviour_labels)
    ax.tick_params(axis='x', labelsize=9)
    
    current_xlim = ax.get_xlim()
    ax.set_xlim(current_xlim[0], current_xlim[1] + extra_whitespace)
    
    ax.legend(loc=legend_loc, fontsize=10, title_fontsize=11)

    # Add a grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.title.set_fontsize(16)
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)

    # Adjust layout and display
    plt.ylim(0.5, 1)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)
    plt.show()
