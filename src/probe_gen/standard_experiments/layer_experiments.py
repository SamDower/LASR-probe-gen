import matplotlib.pyplot as plt

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from probe_gen.config import ConfigDict
import probe_gen.probes as probes


def run_layer_experiments(probe_type, dataset_name, activations_model, probe_C, layers_list, use_bias_options=[True,False], normalize_options=[True,False]):
    """
    Trains a probe on each layer's activations and plots accuracy and roc_auc for each layer.
    Args:
        probe (Probe): An initialised probe to train on each layer's activations.
        repo_id (str): Huggingface repository id.
        activations_filename (str): Huggingface file name for activations.
        labels_filename (str): Labels filename e.g. on_policy_raw.jsonl.
    """
    for layer in layers_list:
        print(f"######################### Evaluating layer {layer} #############################")
        print("Loading from HuggingFace...")
        activations_tensor, attention_mask, labels_tensor = probes.load_hf_activations_and_labels_at_layer(dataset_name, layer)
        print("Aggregating activations...")
        activations_tensor = probes.MeanAggregation()(activations_tensor, attention_mask)
        print("Constructing datasets...")
        train_dataset, val_dataset, test_dataset = probes.create_activation_datasets(activations_tensor, labels_tensor, val_size=0.1, test_size=0.2, balance=True)
        print("Complete.")

        for use_bias in use_bias_options:
            for normalize in normalize_options:

                if probe_type == 'mean':
                    probe = probes.SklearnLogisticProbe(ConfigDict(use_bias=use_bias, C=probe_C, normalize=normalize))
                else:
                    print("Probe type not valid.")
                    return

                # Fit the probe with the datasets
                probe.fit(train_dataset, None)

                # Evaluate the model
                eval_dict, _, _ = probe.eval(val_dataset)
                
                probes.wandb_interface.save_probe_dict_results(
                    eval_dict=eval_dict, 
                    train_set_name=dataset_name,
                    test_set_name=dataset_name,
                    activations_model=activations_model,
                    probe_type=probe_type,
                    hyperparams=[layer, use_bias, normalize, probe_C],
                )
