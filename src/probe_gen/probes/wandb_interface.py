import pandas as pd

import wandb


def save_probe_dict_results(eval_dict, probe_type, probe_use_bias, probe_normalize, probe_C, layer, train_set_name, test_set_name):
    """
    Saves the evaluation dict to wandb as a single run.

    Args:
        eval_dict (dict): evalualtion dictionary obtained from `probe.eval(test_dataset)`.
        probe_type (str): The type of probe trained (e.g. 'mean', 'attention').
        probe_use_bias (bool): Whether use_bias is turned on for the probe.
        probe_normalize (bool): Whether the inputs to the probe are normalized or not. 
        probe_C (float): The inverse of the regularization strength that was used to train the probe.
        layer (int): The layer of the activations the probe was trained on.
        train_set_name (str): An identifiable name for the data the probe was trained on (e.g. refusal_off_other_model).
        test_set_name (str): An identifiable name for the data the probe was tested on (e.g. refusal_on).
        activations_model (str): The model the activations came from.

    Returns:
        None
    """
    # Initialize wandb run
    wandb.init(
        project="LASR_probe_gen",
        entity="samdower",
        config={
            "probe/type": probe_type,
            "probe/use_bias": probe_use_bias,
            "probe/normalize": probe_normalize,
            "probe/C": probe_C,
            "layer": layer,
            "train_dataset": train_set_name,
            "test_dataset": test_set_name,
            "activations_model": activations_model
        }
    )

    # Log metrics
    wandb.log({
        "accuracy": eval_dict['accuracy'],
        "roc_auc": eval_dict['roc_auc'],
        "tpr_at_1_fpr": eval_dict['tpr_at_1_fpr'],
    })

    # Finish the run
    wandb.finish()


def load_probe_eval_dict_by_dict(lookup_dict):
    """
    Loads the latest probe evaluation dictionary which used the dataset names provided.

    Args:
        train_dataset_name (str): An identifiable name for the data the probe was trained on (e.g. refusal_off_other_model).
        test_dataset_name (str): An identifiable name for the data the probe was tested on (e.g. refusal_on).
    
    Returns:
        eval_dict (dict): evalualtion dictionary.
    """
    api = wandb.Api()
    
    # Query runs with specific config filters
    runs = api.runs(
        "samdower/LASR_probe_gen",
        filters=lookup_dict
    )
    
    results = []
    for run in runs:
        accuracy = run.summary.get('accuracy', None)
        roc_auc = run.summary.get('roc_auc', None)
        tpr_at_1_fpr = run.summary.get('tpr_at_1_fpr', None)
        if accuracy is not None:
            results.append({
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'tpr_at_1_fpr': tpr_at_1_fpr
            })
    
    if len(results) == 0:
        print(f"### WARNING ###: could not find run for lookup dict {lookup_dict}, returning None.")
        return None
    elif len(results) > 1:
        print(f"### WARNING ###: multiple runs for lookup dict {lookup_dict}, returning latest.")
    return results[-1]










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
            run_info[attr] = value
    
    # 2. All config parameters with proper key naming
    for key, value in run.config.items():
        # Replace problematic characters in column names
        clean_key = f"config_{key.replace('/', '_').replace('.', '_')}"
        run_info[clean_key] = value
    
    # 3. All summary metrics (final logged values)
    for key, value in run.summary.items():
        # Skip internal wandb keys that start with underscore
        if not key.startswith('_'):
            clean_key = f"metric_{key.replace('/', '_').replace('.', '_')}"
            run_info[clean_key] = value
    
    # 4. System info (if available)
    if hasattr(run, 'system_metrics') and run.system_metrics:
        for key, value in run.system_metrics.items():
            clean_key = f"system_{key.replace('/', '_').replace('.', '_')}"
            run_info[clean_key] = value
    
    return run_info

def load_probe_eval_dicts_as_df(lookup_dict):
    """
    Loads the latest probe evaluation dictionary which used the dataset names provided.

    Args:
        train_dataset_name (str): An identifiable name for the data the probe was trained on (e.g. refusal_off_other_model).
        test_dataset_name (str): An identifiable name for the data the probe was tested on (e.g. refusal_on).
    
    Returns:
        eval_dict (dict): evalualtion dictionary.
    """
    api = wandb.Api()
    
    # Query runs with specific config filters
    runs = api.runs(
        "samdower/LASR_probe_gen",
        filters=lookup_dict
    )
    
    results = []
    for run in runs:        
        # Get basic info
        run_info = extract_all_run_info(run)
        
        results.append(run_info)
    
    df = pd.DataFrame(results)
    return df