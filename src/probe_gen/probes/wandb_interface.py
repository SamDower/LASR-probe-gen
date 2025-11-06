import json
import os

import pandas as pd

import wandb


def save_probe_dict_results(eval_dict, probe_type, behaviour, train_set, test_set, activations_model, hyperparams):
    """
    Saves the evaluation dict to wandb as a single run.
    Args:
        eval_dict (dict): evalualtion dictionary obtained from `probe.eval(test_dataset)`.
        probe_type (str): The type of probe trained (e.g. 'mean', 'attention').
        behaviour (str): The behaviour the probe was trained and tested on.
        train_set_name (list): An identifiable list for the data the probe was trained on:
            - [datasource, generation_method, response_model]
        test_set_name (list): An identifiable list for the data the probe was tested on:
            - [datasource, generation_method, response_model]
        activations_model (str): The model the activations came from.
        hyperparams (list): A list of hyperparams used to train the probe. The order of the hyperparams:
            - for "attention_torch": [layer, use_bias, normalize, lr, weight_decay]
            - for "mean_torch": [layer, use_bias, normalize, lr, weight_decay]
            - for "mean": [layer, use_bias, normalize, C]
    """
    os.environ["WANDB_SILENT"] = "true"

    # Initialize wandb run
    config_dict = {
        "probe/type": probe_type,
        "behaviour": behaviour,
        "train/datasource": train_set[0],
        "train/generation_method": train_set[1],
        "train/response_model": train_set[2],
        "test/datasource": test_set[0],
        "test/generation_method": test_set[1],
        "test/response_model": test_set[2],
        "activations_model": activations_model,
        "layer": hyperparams[0],
        "probe/use_bias": hyperparams[1],
        "probe/normalize": hyperparams[2],
    }
    if 'torch' in probe_type:
        config_dict["probe/lr"] = hyperparams[3]
        config_dict["probe/weight_decay"] = hyperparams[4]
    elif probe_type == "mean":
        config_dict["probe/C"] = hyperparams[3]
    else:
        print("Probe type not valid.")
        return
    
    wandb.init(
        project="LASR_probe_gen",
        entity="LasrProbeGen",
        config=config_dict,
    )
    # Log metrics
    wandb.log({
        "accuracy": None if 'accuracy' not in eval_dict else eval_dict['accuracy'],
        "roc_auc": None if 'roc_auc' not in eval_dict else eval_dict['roc_auc'],
        "tpr_at_1_fpr": None if 'tpr_at_1_fpr' not in eval_dict else eval_dict['tpr_at_1_fpr'],
    })
    # Finish the run
    wandb.finish()
    # print("Saved run")


def load_probe_eval_dict_by_dict(lookup_dict):
    """
    Loads the latest probe evaluation dictionary which used the dataset names provided.

    Args:
        train_dataset_name (str): An identifiable name for the data the probe was trained on (e.g. refusal_off_other_model).
        test_dataset_name (str): An identifiable name for the data the probe was tested on (e.g. refusal_on).
    
    Returns:
        eval_dict (dict): evalualtion dictionary.
    """
    os.environ["WANDB_SILENT"] = "true"

    api = wandb.Api()
    
    # Query runs with specific config filters
    runs = api.runs(
        "LasrProbeGen/LASR_probe_gen",
        filters=lookup_dict
    )
    
    results = []
    for run in runs:
        data_dict = run.summary._json_dict
        accuracy = data_dict.get('accuracy', None)
        roc_auc = data_dict.get('roc_auc', None)
        tpr_at_1_fpr = data_dict.get('tpr_at_1_fpr', None)
        if roc_auc is not None:
            results.append({
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'tpr_at_1_fpr': tpr_at_1_fpr
            })
    
    if len(results) == 0:
        print(f"### WARNING ###: could not find run for lookup dict {lookup_dict}, returning None.")
        return None
    elif len(results) > 1:
        # print(f"### WARNING ###: multiple runs for lookup dict {lookup_dict}, returning latest.")
        pass
    return results[-1]


# def _extract_common_filters(lookup_dicts_list):
#     """
#     Extract key-value pairs that are common to ALL dictionaries in the list.
#     These can be used as filters in the wandb API call without losing any data.
    
#     Args:
#         lookup_dicts_list (list): List of lookup dictionaries
    
#     Returns:
#         common_filters (dict): Dictionary containing key-value pairs common to all lookups
#         remaining_lookups (list): List of dictionaries with common keys removed
#     """
#     if not lookup_dicts_list:
#         return {}, []
    
#     if len(lookup_dicts_list) == 1:
#         # If only one dict, all keys are "common"
#         return lookup_dicts_list[0].copy(), [{}]
    
#     # Find keys that exist in ALL dictionaries
#     common_keys = set(lookup_dicts_list[0].keys())
#     for lookup_dict in lookup_dicts_list[1:]:
#         common_keys &= set(lookup_dict.keys())
    
#     # Among common keys, find those with the same value in ALL dictionaries
#     common_filters = {}
#     for key in common_keys:
#         first_value = lookup_dicts_list[0][key]
#         if all(lookup_dict[key] == first_value for lookup_dict in lookup_dicts_list):
#             common_filters[key] = first_value
    
#     # Create remaining lookup dicts with common filters removed
#     remaining_lookups = []
#     for lookup_dict in lookup_dicts_list:
#         remaining_dict = {k: v for k, v in lookup_dict.items() 
#                          if k not in common_filters}
#         remaining_lookups.append(remaining_dict)
    
#     return common_filters


# def load_probe_eval_dict_batch(lookup_dicts_list):
#     """
#     Loads probe evaluation dictionaries for multiple lookup criteria with a single API call.
    
#     Args:
#         lookup_dicts_list (list): List of lookup dictionaries, each containing config filters
    
#     Returns:
#         results_dict (dict): Dictionary mapping lookup_dict (as tuple of sorted items) to evaluation results
#     """
#     os.environ["WANDB_SILENT"] = "true"
#     api = wandb.Api()
    
#     common_filters = _extract_common_filters(lookup_dicts_list)
    
#     # Get all runs from the project with a single API call
#     all_runs = api.runs("LasrProbeGen/LASR_probe_gen", filters=common_filters)

#     # Convert lookup dicts to tuples for use as dictionary keys
#     lookup_tuples = [tuple(sorted(lookup_dict.items())) for lookup_dict in lookup_dicts_list]
    
#     # Initialize results for each lookup dict
#     results_dict = {}
#     for lookup_tuple in lookup_tuples:
#         results_dict[lookup_tuple] = []
    
#     # Filter runs locally
#     for run in all_runs:
#         run_config = run.config
#         if isinstance(run_config, str):
#             run_config = json.loads(run_config)
            
#         # Check each lookup dict to see if this run matches
#         for i, lookup_dict in enumerate(lookup_dicts_list):
#             lookup_tuple = lookup_tuples[i]
            
#             # Check if all key-value pairs in lookup_dict match the run's config
#             if all(not key.startswith("config.") or run_config.get(key[7:]) == value for key, value in lookup_dict.items()):
#                 # accuracy = run.summary.get('accuracy', None)
#                 roc_auc = run.summary.get('roc_auc', None)
#                 # tpr_at_1_fpr = run.summary.get('tpr_at_1_fpr', None)
                
#                 if roc_auc is not None:
#                     results_dict[lookup_tuple].append({
#                         # 'accuracy': accuracy,
#                         'roc_auc': roc_auc,
#                         # 'tpr_at_1_fpr': tpr_at_1_fpr
#                     })
    
#     # Process results similar to original function (take latest if multiple)
#     final_results = {}
#     for i, lookup_dict in enumerate(lookup_dicts_list):
#         lookup_tuple = lookup_tuples[i]
#         results = results_dict[lookup_tuple]
        
#         if len(results) == 0:
#             print(f"### WARNING ###: could not find run for lookup dict {lookup_dict}, returning None.")
#             final_results[lookup_tuple] = None
#         elif len(results) > 1:
#             # Optionally uncomment the warning below
#             # print(f"### WARNING ###: multiple runs for lookup dict {lookup_dict}, returning latest.")
#             final_results[lookup_tuple] = results[-1]
#         else:
#             final_results[lookup_tuple] = results[0]
    
#     results_list = []
#     for lookup_dict in lookup_dicts_list:
#         lookup_tuple = tuple(sorted(lookup_dict.items()))
#         results_list.append(final_results[lookup_tuple])
    
#     return results_list


def unwrap_value(x):
    """If W&B stores {'value': ...}, unwrap it. Otherwise return unchanged."""
    if isinstance(x, dict) and "value" in x and len(x) == 1:
        return x["value"]
    return x



def extract_all_run_info(run):
    """
    Automatically extract all available information from a wandb run
    """
    run_info = {}
    
    # 1. Basic run metadata - extract all available attributes
    basic_attrs = ['id', 'name', 'state', 'created_at', 'updated_at', 'url', 'path', 
                   'notes', 'tags', 'group', 'job_type', 'sweep', 'project', 'entity']
    
    for attr in basic_attrs:
        if hasattr(run, attr):
            value = getattr(run, attr)
            # Convert lists/complex objects to strings for DataFrame compatibility
            if isinstance(value, (list, dict)):
                value = str(value) if value else None
            run_info[attr] = value["value"] if isinstance(value, dict) else value
    
    # 2. All config parameters with proper key naming
    run_config = run.config
    if isinstance(run_config, str):
        run_config = json.loads(run_config)
    for key, value in run_config.items():
        # Replace problematic characters in column names
        clean_key = f"config_{key.replace('/', '_').replace('.', '_')}"
        run_info[clean_key] = value["value"] if isinstance(value, dict) else value
    
    # 3. All summary metrics (final logged values)
    run_summary = run.summary
    if hasattr(run_summary, "_json_dict") and isinstance(run_summary._json_dict, str):
        run_summary = json.loads(run_summary._json_dict)
    for key, value in run_summary.items():
        # Skip internal wandb keys that start with underscore
        if not key.startswith('_'):
            clean_key = f"metric_{key.replace('/', '_').replace('.', '_')}"
            run_info[clean_key] = value["value"] if isinstance(value, dict) else value
    
    # 4. System info (if available)
    if hasattr(run, 'system_metrics') and run.system_metrics:
        run_system_metrics = run.system_metrics
        if isinstance(run_system_metrics, str):
            run_system_metrics = json.loads(run_system_metrics)
        for key, value in run_system_metrics.items():
            clean_key = f"system_{key.replace('/', '_').replace('.', '_')}"
            run_info[clean_key] = value["value"] if isinstance(value, dict) else value
    
    return run_info


def load_probe_eval_dicts_as_df(lookup_dict):
    api = wandb.Api()
    
    # Query runs with specific config filters
    runs = api.runs(
        "LasrProbeGen/LASR_probe_gen",
        filters=lookup_dict
    )
    
    results = []
    for run in runs:        
        # Get basic info
        run_info = extract_all_run_info(run)
        results.append(run_info)
    
    df = pd.DataFrame(results)
    return df