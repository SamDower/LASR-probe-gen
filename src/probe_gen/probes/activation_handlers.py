import json
import os

import joblib
import torch
from huggingface_hub import hf_hub_download
from torch.nn.utils.rnn import pad_sequence

from probe_gen.config import data


def _download_labels_from_hf(repo_id, labels_filepath, labels_localpath):
    _ = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=labels_filepath,
        local_dir=labels_localpath,
        token=os.environ.get("HF_TOKEN")
    )
    

def _load_labels_from_local_jsonl(labels_filename, verbose=False):
    labels_list = []
    with open(labels_filename, 'r') as file:
        for line in file:
            data = json.loads(line)
            if data["scale_labels"] <= 5:
                labels_list.append(1.0)
            else:
                labels_list.append(0.0)
    labels_tensor = torch.tensor(labels_list)
    if verbose: 
        print("loaded labels")
    return labels_tensor


def _load_activations_from_hf(repo_id, filename, verbose=False):
    # Load activations
    file_path = hf_hub_download(
        repo_id=repo_id, 
        filename=filename,
        repo_type="dataset",
    )
    df = joblib.load(file_path)

    # Extract all activations
    all_activations = []
    for i in range(len(df)):
        all_activations.append(df.loc[i]["activations"])
    activations_tensor = pad_sequence(all_activations, batch_first=True, padding_value=0.0).to(torch.float32)
    if verbose: 
        print(f"loaded activations with shape {activations_tensor.shape}")

    max_len = activations_tensor.shape[1]
    masks = []
    for tensor in all_activations:
        current_len = tensor.shape[0]
        mask = torch.ones(1, current_len)
        if current_len < max_len:
            padding_mask = torch.zeros(1, max_len - current_len)
            mask = torch.cat([mask, padding_mask], dim=1)
        masks.append(mask)
    attention_mask = torch.cat(masks, dim=0)
    if verbose: 
        print(f"calculated attention mask with shape {attention_mask.shape}")

    return activations_tensor, attention_mask


def load_hf_activations_at_layer(
    behaviour: str, 
    datasource: str, 
    activations_model: str="llama_3b", 
    response_model: str="llama_3b", 
    generation_method: str="on_policy", 
    mode: str="train", 
    layer: int=12, 
    and_labels: bool=False, 
    verbose: bool=False):
    """
    Loads activations for a specified layer and ground truth labels from Huggingface.

    Args:
        behaviour (str): Behaviour name.
        datasource (str): Datasource name.
        activations_model (str): Activations model name.
        response_model (str): Response model name.
        generation_method (str): "on_policy", "prompted", "incentivised", "off_policy".
        mode (str): "train" or "test".
        layer (int): Model layer we should get the activations from. 
        and_labels (bool): Whether to load labels.
        verbose (bool): Should the function output to console. 
    
    Returns:
        activations_tensor (tensor): tensor of activations of shape [batch_size, seq_len, dim].
        attention_mask (tensor): tensor stating which tokens are real (1) or padded (0) of shape [batch_size, seq_len]
        labels_tensor (tensor): tensor of ground truth labels of shape [batch_size].
    """
    if datasource == "trading":
        mode = "3.5k"
        
    try:
        repo_id = f"lasrprobegen/{behaviour}-activations"
        filepath = f"{datasource}/{activations_model}/{response_model}_{generation_method}_{mode}_layer_{layer}.pkl"
        activations_tensor, attention_mask = _load_activations_from_hf(repo_id, filepath, verbose)
    except Exception as e:
        # Try loading with on_policy in the name instead
        if generation_method == "off_policy":
            real_generation_method = "incentivised" if behaviour in ["deception", "sandbagging"] else "on_policy"
            filepath = filepath.replace("off_policy", real_generation_method)
            activations_tensor, attention_mask = _load_activations_from_hf(repo_id, filepath, verbose)
        else:
            raise e
        
    if and_labels:
        generation_method_for_labels = generation_method.replace("_included", "")
        labels_filepath = f"{datasource}/{response_model}_{generation_method_for_labels}_{mode}.jsonl"
        labels_localpath = data.data / behaviour
        try:
            if not os.path.exists(labels_localpath / labels_filepath):
                _download_labels_from_hf(repo_id, labels_filepath, labels_localpath)
            labels_tensor = _load_labels_from_local_jsonl(labels_localpath / labels_filepath, verbose)
        except Exception as e:
            # Try loading with on_policy in the name instead
            if generation_method == "off_policy":
                real_generation_method = "incentivised" if behaviour in ["deception", "sandbagging"] else "on_policy"
                labels_filepath = labels_filepath.replace("off_policy", real_generation_method)
                if not os.path.exists(labels_localpath / labels_filepath):
                    _download_labels_from_hf(repo_id, labels_filepath, labels_localpath)
                labels_tensor = _load_labels_from_local_jsonl(labels_localpath / labels_filepath, verbose)
            else:
                raise e
        return activations_tensor, attention_mask, labels_tensor
    
    else:
        return activations_tensor, attention_mask

    
def create_activation_datasets(activations_tensor, labels_tensor, splits=[3500, 500, 0], verbose=False):
    """
    Create datasets from pre-aggregated activations.

    Args:
        activations_tensor (tensor): tensor of pre-aggregated activations of shape [batch_size, dim] 
        labels_tensor (tensor): tensor of ground truth labels of shape [batch_size]
        verbose (bool): Should the function output to console. 
    """
    torch.manual_seed(0)

    if len(splits) != 3:
        raise ValueError("Splits must be a list of 3 numbers [train_size, val_size, test_size]")
    
    if sum(splits) > labels_tensor.shape[0]:
        if (sum(splits) - (labels_tensor.shape[0])) > 500:
            raise ValueError("Splits must sum to less than or equal to number of samples, within a margin of 500")
            
        # Keep the val and test sizes the same but reduce the train size
        val_test_size = splits[1] + splits[2]
        train_size = labels_tensor.shape[0] - val_test_size
        print(f"Do not have {splits[0]} training samples, using {train_size} instead")
        splits[0] = train_size
    
    # Get indices for each label and subsample both classes to same size
    label_0_indices = (labels_tensor == 0.0).nonzero(as_tuple=True)[0]
    label_1_indices = (labels_tensor == 1.0).nonzero(as_tuple=True)[0]
    min_class_count = min(len(label_0_indices), len(label_1_indices))
    label_0_indices = label_0_indices[:min_class_count]
    label_1_indices = label_1_indices[:min_class_count]        

    # Compute split sizes (divided by 2 because we have two classes)
    n_train = splits[0] // 2
    n_val = splits[1] // 2
    n_test = splits[2] // 2

    # Split label 0s
    train_0 = label_0_indices[:n_train]
    val_0 = label_0_indices[n_train:n_train + n_val]
    test_0 = label_0_indices[n_train + n_val:n_train + n_val + n_test]

    # Split label 1s
    train_1 = label_1_indices[:n_train]
    val_1 = label_1_indices[n_train:n_train + n_val]
    test_1 = label_1_indices[n_train + n_val:n_train + n_val + n_test]

    # Concatenate splits and shuffle within each
    def get_split(indices_0, indices_1):
        indices = torch.cat([indices_0, indices_1])
        indices = indices[torch.randperm(len(indices))]
        return activations_tensor[indices], labels_tensor[indices]

    X_train, y_train = get_split(train_0, train_1)
    X_val, y_val = get_split(val_0, val_1)
    X_test, y_test = get_split(test_0, test_1)

    # Output balance
    if verbose:
        print(f"Train: {y_train.shape[0]} samples, {y_train.sum()} positives")
        print(f"Val:   {y_val.shape[0]} samples, {y_val.sum()} positives")
        print(f"Test:  {y_test.shape[0]} samples, {y_test.sum()} positives")
    
    train_dataset = {'X': X_train, 'y': y_train}
    val_dataset = {'X': X_val, 'y': y_val}
    if n_val == 0.0:
        val_dataset = None
    test_dataset = {'X': X_test, 'y': y_test}

    return train_dataset, val_dataset, test_dataset


    