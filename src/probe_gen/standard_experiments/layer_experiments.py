import matplotlib.pyplot as plt

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
import probes


def run_layer_experiments(probe, repo_id, activations_filename, labels_filename):
    """
    Trains a probe on each layer's activations and plots accuracy and roc_auc for each layer.

    Args:
        probe (Probe): An initialised probe to train on each layer's activations.
        repo_id (str): Huggingface repository id.
        activations_filename (str): Huggingface file name for activations.
        labels_filename (str): Labels filename e.g. on_policy_raw.jsonl.
    
    Returns:
        accuracies (list): a list of accuracies for each layer.
        roc_aucs (list): a list of roc_aucs for each layer.
    """

    accuracies = []
    roc_aucs = []

    print("loading activations for each layer (may take ~ 1 minute)")
    activations_tensors, attention_mask, labels_tensor = probes.load_hf_activations_and_labels(repo_id, activations_filename, labels_filename)

    for layer in activations_tensors.keys():
        print(f"######################### Evaluating layer {layer} #############################")

        activations_tensor = probes.MeanAggregation()(activations_tensors[layer], attention_mask)
        train_dataset, val_dataset, test_dataset = probes.create_activation_datasets(activations_tensor, labels_tensor, val_size=0, test_size=0.2, balance=True)

        # Fit the probe with the datasets
        probe.fit(train_dataset, val_dataset, normalize=True)

        # Evaluate the model
        eval_dict, _, _ = probe.eval(test_dataset)
        accuracies.append(eval_dict['accuracy'])
        roc_aucs.append(eval_dict['roc_auc'])


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot accuracies
    ax1.plot(activations_tensors.keys(), accuracies, marker='o', linewidth=2, markersize=6)
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True, alpha=0.3)

    # Plot ROC AUCs
    ax2.plot(activations_tensors.keys(), roc_aucs, marker='s', linewidth=2, markersize=6)
    ax2.set_title('ROC AUC')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('ROC AUC')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.legend()
    plt.show()

    return accuracies, roc_aucs