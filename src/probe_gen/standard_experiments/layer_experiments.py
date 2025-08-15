import matplotlib.pyplot as plt

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
import probes


def run_layer_experiments(probe_type, dataset_name, layers_to_run, use_bias_options=[True,False], normalize_inputs_options=[True,False]):
    """
    Trains a probe on each layer's activations and plots accuracy and roc_auc for each layer.

    Args:
        probe (Probe): An initialised probe to train on each layer's activations.
        repo_id (str): Huggingface repository id.
        activations_filename (str): Huggingface file name for activations.
        labels_filename (str): Labels filename e.g. on_policy_raw.jsonl.
    
    Returns:
        None
    """

    for layer in layers_to_run:
        print(f"######################### Evaluating layer {layer} #############################")

        print("loading activations for each layer (may take ~ 1 minute)")
        activations_tensor, attention_mask, labels_tensor = probes.load_hf_activations_and_labels_at_layer(dataset_name, layer)
        train_dataset, val_dataset, test_dataset = probes.create_activation_datasets(activations_tensor, attention_mask, labels_tensor, val_size=0, test_size=0.2)

        for use_bias in use_bias_options:
            for normalize_inputs in normalize_inputs_options:

                if probe_type == 'mean':
                    probe = probes.SklearnMeanLogisticProbe(use_bias=use_bias)
                else:
                    print("Probe type not valid.")
                    probe = None

                # Fit the probe with the datasets
                probe.fit(train_dataset, val_dataset, normalize=normalize_inputs)

                # Evaluate the model
                eval_dict, _, _ = probe.eval(test_dataset)

                probes.wandb_interface.save_probe_dict_results(
                    eval_dict, 
                    probe_type, 
                    use_bias, 
                    normalize_inputs, 
                    layer, 
                    dataset_name,
                    dataset_name
                )
