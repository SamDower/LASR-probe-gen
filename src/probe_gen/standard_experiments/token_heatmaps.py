import json

import pandas as pd

# Load the JSONL file that contains questions, model outputs, and labels from HuggingFace
from huggingface_hub import hf_hub_download

import probe_gen.probes as probes
from probe_gen.config import ConfigDict


def train_probe_on(probe_type, behaviour, datasource, activations_model, generation_method, response_model, mode, verbose=False):

    # Load the best hyperparameters or set your own
    cfg = ConfigDict.from_json(activations_model, probe_type, behaviour)
    if verbose:
        print("Loaded hyperparameters:")
        print(f"  Layer: {cfg.layer}")
        print(f"  Use bias: {cfg.use_bias}")
        print(f"  Normalize: {cfg.normalize}")
        if hasattr(cfg, 'C'):
            print(f"  C (inverse ofregularization): {cfg.C}")

    # Load activations and labels from HuggingFace
    activations_tensor, attention_mask, labels_tensor = probes.load_hf_activations_at_layer(
        behaviour, 
        datasource, 
        activations_model, 
        response_model, 
        generation_method, 
        mode, 
        cfg.layer, 
        and_labels=True, 
        verbose=verbose
    )

    if verbose:
        print("\nLoaded data:")
        print(f"  Activations shape: {activations_tensor.shape}")
        print(f"  Attention mask shape: {attention_mask.shape}")
        print(f"  Labels shape: {labels_tensor.shape}")
        print(f"  Positive samples: {labels_tensor.sum().item():.0f}")
        print(f"  Negative samples: {(len(labels_tensor) - labels_tensor.sum()).item():.0f}")


    # Aggregate activations if using mean probe
    if "mean" in probe_type:
        activations_aggregated = probes.MeanAggregation()(activations_tensor, attention_mask)
        if verbose:
            print(f"Aggregated activations shape: {activations_aggregated.shape}")
    else:
        activations_aggregated = activations_tensor
        if verbose:
            print(f"Using full sequence activations: {activations_aggregated.shape}")

    # Create train and validation datasets
    split_val = 2500 if behaviour in ["deception", "sandbagging"] else 3500

    train_dataset, val_dataset, test_dataset = probes.create_activation_datasets(
        activations_aggregated, labels_tensor, splits=[split_val, 500, 0], verbose=True)

    # Initialize probe
    if probe_type == "mean":
        probe = probes.SklearnLogisticProbe(cfg)
    elif probe_type == "mean_torch":
        probe = probes.TorchLinearProbe(cfg)
    elif probe_type == "attention_torch":
        probe = probes.TorchAttentionProbe(cfg)

    # Train the probe
    if verbose:
        print("\nTraining probe...")
    probe.fit(train_dataset, val_dataset)

    # Evaluate on validation set
    if verbose:
        eval_dict, y_pred, y_pred_proba = probe.eval(val_dataset)
        print(f'\n✓ Validation ROC-AUC: {eval_dict["roc_auc"]:.4f}')
        print(f'  Accuracy: {eval_dict["accuracy"]:.4f}')
    
    return probe



def extract_probe_weights(probe):
    # Extract probe weights
    if isinstance(probe, probes.SklearnLogisticProbe):
        # For sklearn probe
        probe_weights = probe.classifier.coef_[0]  # Shape: (hidden_dim,)
        probe_bias = probe.classifier.intercept_[0]        
    elif isinstance(probe, probes.TorchLinearProbe):
        # For PyTorch mean probes
        probe_weights = probe.model.linear.weight.detach().cpu().numpy()[0]  # Shape: (hidden_dim,)
        probe_bias = probe.model.linear.bias.detach().cpu().numpy()[0]
    elif isinstance(probe, probes.TorchAttentionProbe):
        # For attention probe, get the query/key/value weights
        probe_weights = {
            'W_Q': probe.model.W_Q.detach().cpu().numpy(),
            'W_K': probe.model.W_K.detach().cpu().numpy(),
            'W_V': probe.model.W_V.detach().cpu().numpy(),
            'W_out': probe.model.W_out.detach().cpu().numpy()
        }
        probe_bias = None
    else:
        raise ValueError(f"Unsupported probe type: {type(probe)}")

    return probe_weights, probe_bias



def load_labelled_responses_and_activations(probe_type, behaviour, datasource, activations_model, generation_method, response_model, mode, verbose=False):

    generation_method_for_labels = generation_method.replace("_included", "")
    if generation_method == "off_policy":
        generation_method = "on_policy"
    labels_filename = f"{datasource}/{response_model}_{generation_method_for_labels}_{mode}.jsonl"

    # Download from HuggingFace
    repo_id = f"lasrprobegen/{behaviour}-activations"

    labels_localpath = hf_hub_download(
        repo_id=repo_id,
        filename=labels_filename,
        repo_type="dataset"
    )

    # Load the best hyperparameters or set your own
    cfg = ConfigDict.from_json(activations_model, probe_type, behaviour)

    # Load activations and labels from HuggingFace
    activations_tensor, attention_mask, labels_tensor = probes.load_hf_activations_at_layer(
        behaviour, 
        datasource, 
        activations_model, 
        response_model, 
        generation_method, 
        mode, 
        cfg.layer, 
        and_labels=True, 
        verbose=verbose
    )

    # Load the JSONL file
    data_rows = []
    with open(labels_localpath, 'r') as file:
        for line in file:
            data_dict = json.loads(line)
            data_rows.append(data_dict)

    # Convert to DataFrame for easier manipulation
    dataset_df = pd.DataFrame(data_rows)

    # if 'input' in dataset_df.columns:
    #     print(f"  Input: {dataset_df.iloc[0]['input'][:200]}...")
    # if 'model_outputs' in dataset_df.columns:
    #     print(f"  Output: {dataset_df.iloc[0]['model_outputs'][:200]}...")
    # if 'scale_labels' in dataset_df.columns:
    #     print(f"  Label: {dataset_df.iloc[0]['scale_labels']}")
    
    return dataset_df, activations_tensor, attention_mask, labels_tensor