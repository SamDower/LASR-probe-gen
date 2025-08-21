from types import NoneType
import matplotlib.pyplot as plt

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from notebooks.TrainProbe import best_roc
import probes
from probes.wandb_interface import load_probe_eval_dict_by_dict, load_probe_eval_dicts_as_df


def run_full_hyp_search_on_layers(probe_type, dataset_name, activations_model, layers_to_run):
    """
    Trains a probe on each layer's activations and plots accuracy and roc_auc for each layer.

    Args:
        probe_type (str): The type of probe to train (e.g. 'mean').
        dataset_name (str): The dataset to both train and test on.
        activations_filename (str): Huggingface file name for activations.
        labels_filename (str): Labels filename e.g. on_policy_raw.jsonl.
    
    Returns:
        None
    """

    best_roc_auc = 0
    best_roc_auc_params = {}

    for layer in layers_to_run:
        print(f"######################### Evaluating layer {layer} #############################")

        print("loading activations (may take ~ 1 minute)")
        activations_tensor, attention_mask, labels_tensor = probes.load_hf_activations_and_labels_at_layer(dataset_name, layer)
        activations_tensor = probes.MeanAggregation()(activations_tensor, attention_mask)
        train_dataset, val_dataset, test_dataset = probes.create_activation_datasets(activations_tensor, labels_tensor, val_size=0.1, test_size=0.2, balance=True)

        for use_bias in [True, False]:
            for normalize_inputs in [True, False]:
                for C in [0.001, 0.01, 0.1, 1, 10]:

                    if probe_type == 'mean':
                        probe = probes.SklearnLogisticProbe(use_bias=use_bias, C=C, normalize=normalize_inputs)
                    else:
                        print("Probe type not valid.")
                        probe = None

                    # Fit the probe with the datasets
                    probe.fit(train_dataset, None)

                    # Evaluate the model
                    eval_dict, _, _ = probe.eval(val_dataset)

                    probes.wandb_interface.save_probe_dict_results(
                        eval_dict, 
                        probe_type, 
                        use_bias, 
                        normalize_inputs, 
                        C,
                        layer, 
                        dataset_name,
                        dataset_name,
                        activations_model
                    )


def load_best_params_from_search(probe_type, dataset_name, activations_model, layers_list):

    df = load_probe_eval_dicts_as_df({
        "config.probe_type": probe_type,
        "config.train_dataset": dataset_name,
        "config.test_dataset": dataset_name,
        "config.activations_model": activations_model,
        "state": "finished"  # Only completed runs
    })

    best_roc_auc = 0
    best_roc_auc_params = {}

    for use_bias in [True, False]:
        for normalize_inputs in [True, False]:
            for C in [0.001, 0.01, 0.1, 1, 10]:
                for layer in layers_list:
                    filtered_df = df
                    filtered_df = filtered_df[filtered_df['config_probe_normalize'] == normalize_inputs]
                    filtered_df = filtered_df[filtered_df['config_probe_use_bias'] == use_bias]
                    filtered_df = filtered_df[filtered_df['config_probe_C'] == C]
                    filtered_df = filtered_df[filtered_df['config_layer'] == layer]
                    if filtered_df.shape[0] >= 1:
                        roc_auc = filtered_df['metric_roc_auc'].iloc[0]

                        if roc_auc > best_roc_auc:
                            best_roc_auc = roc_auc
                            best_roc_auc_params = {
                                'layer': layer,
                                'use_bias': use_bias,
                                'normalize': normalize_inputs,
                                'C': C
                            }

    print(f"Best roc_auc: {roc_auc}")
    print(f"Best params: {best_roc_auc_params}")
