import wandb

def save_probe_dict_results(eval_dict, probe_type, probe_use_bias, probe_normalize, train_set_name, test_set_name):
    """
    Saves the evaluation dict to wandb as a single run.

    Args:
        eval_dict (dict): evalualtion dictionary obtained from `probe.eval(test_dataset)`.
        probe_type (str): The type of probe trained (e.g. 'mean', 'attention').
        probe_use_bias (bool): Whether use_bias is turned on for the probe.
        probe_normalize (bool): Whether the inputs to the probe are normalized or not. 
        train_set_name (str): An identifiable name for the data the probe was trained on (e.g. refusal_off_other_model).
        test_set_name (str): An identifiable name for the data the probe was tested on (e.g. refusal_on).
    
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
            "train_dataset": train_set_name,
            "test_dataset": test_set_name
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


def load_probe_eval_dict_by_datasets(train_dataset_name, test_dataset_name):
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
        filters={
            "config.train_dataset": train_dataset_name,
            "config.test_dataset": test_dataset_name,
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
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'tpr_at_1_fpr': tpr_at_1_fpr
            })
    
    if len(results) == 0:
        print(f"### WARNING ###: could not find run for dataset pair ({train_dataset_name}, {test_dataset_name}), returning None.")
        return None
    elif len(results) > 1:
        print(f"### WARNING ###: multiple runs for dataset pair ({train_dataset_name}, {test_dataset_name}), returning latest.")
    return results[-1]