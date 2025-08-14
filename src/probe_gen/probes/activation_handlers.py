import json
import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from huggingface_hub import hf_hub_download


def _create_attention_mask(tensor_list, max_len):
    """
    Healper function to create attention mask: 1 for real tokens, 0 for padding
    """
    masks = []
    for tensor in tensor_list:
        current_len = tensor.shape[0]
        mask = torch.ones(1, current_len)
        if current_len < max_len:
            padding_mask = torch.zeros(1, max_len - current_len)
            mask = torch.cat([mask, padding_mask], dim=1)
        masks.append(mask)
    return torch.cat(masks, dim=0)


# def load_hf_activations_and_labels(repo_id, activations_filename, labels_filename, verbose=False):
#     """
#     Loads activations for all layers and ground truth labels from Huggingface.

#     Args:
#         repo_id (str): Huggingface repository id.
#         activations_filename (str): Huggingface file name for activations.
#         labels_filename (str): Labels filename e.g. on_policy_raw.jsonl.
#         verbose (bool): Should the function output to console. 
    
#     Returns:
#         activations_tensors (dict): dictionary of tensors of activations, each with shape [batch_size, seq_len, dim] with keys as ints for each layer.
#         attention_mask (tensor): tensor stating which tokens are real (1) or padded (0) of shape [batch_size, seq_len]
#         labels_tensor (tensor): tensor of ground truth labels of shape [batch_size].
#     """
#     # Load the labels (TODO: currently from the json file but we will need to change this to load from huggingface as well)
#     labels_list = []
#     with open(labels_filename, 'r') as file:
#         for line in file:
#             data = json.loads(line)
#             if data["scale_labels"] <= 5:
#                 labels_list.append(1.0)
#             else:
#                 labels_list.append(0.0)
#     labels_tensor = torch.tensor(labels_list)
#     if verbose: print("loaded labels")

#     # Load activations
#     file_path = hf_hub_download(repo_id=repo_id, filename=activations_filename, repo_type="dataset")
#     df = pd.read_pickle(file_path)

#     activations_tensors = {}
#     attention_mask = None
#     # Extract all activations
#     for layer in df.loc[0]["activations"].keys():
#         all_activations = []
#         for i in range(len(df)):
#             all_activations.append(df.loc[i]["activations"][layer])
#         activations_tensor = pad_sequence(all_activations, batch_first=True, padding_value=0.0).to(torch.float32)
#         activations_tensors[layer] = activations_tensor
#         if attention_mask == None:
#             attention_mask = _create_attention_mask(all_activations, activations_tensor.shape[1])

#     if verbose: print(f"loaded activations")

#     return activations_tensors, attention_mask, labels_tensor


def load_hf_activations_and_labels(repo_id, activations_filename, labels_filename, verbose=False):
    """
    Loads activations for a specified layer and ground truth labels from Huggingface.

    Args:
        repo_id (str): Huggingface repository id.
        activations_filename (str): Huggingface file name for activations.
        labels_filename (str): Labels filename e.g. on_policy_raw.jsonl.
        layer (int): Model layer we should get the activations from. 
        verbose (bool): Should the function output to console. 
    
    Returns:
        activations_tensor (tensor): tensor of activations of shape [batch_size, seq_len, dim].
        attention_mask (tensor): tensor stating which tokens are real (1) or padded (0) of shape [batch_size, seq_len]
        labels_tensor (tensor): tensor of ground truth labels of shape [batch_size].
    """
    # Load the labels (TODO: currently from the json file but we will need to change this to load from huggingface as well)
    labels_list = []
    with open(labels_filename, 'r') as file:
        for line in file:
            data = json.loads(line)
            if data["scale_labels"] <= 5:
                labels_list.append(1.0)
            else:
                labels_list.append(0.0)
    labels_tensor = torch.tensor(labels_list)
    if verbose: print("loaded labels")

    # Load activations
    file_path = hf_hub_download(repo_id=repo_id, filename=activations_filename, repo_type="dataset")
    df = pd.read_pickle(file_path)

    # Extract all activations
    all_activations = []
    for i in range(len(df)):
        all_activations.append(df.loc[i]["activations"])
    activations_tensor = pad_sequence(all_activations, batch_first=True, padding_value=0.0).to(torch.float32)
    if verbose: print(f"loaded activations with shape {activations_tensor.shape}")

    attention_mask = _create_attention_mask(all_activations, activations_tensor.shape[1])
    if verbose: print(f"calculated attention mask with shape {attention_mask.shape}")

    return activations_tensor, attention_mask, labels_tensor












def _train_val_test_split_torch(X, y, val_size=0.1, test_size=0.2):
    """
    Helper function to split data into train, validation, and test sets
    """

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * val_size)
    
    # Create random indices
    indices = torch.randperm(n_samples, device=X.device)
    
    # Split indices
    test_indices = indices[:n_test]
    val_indices = indices[n_test:n_test + n_val]
    train_indices = indices[n_test + n_val:]
    
    # Split data
    X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
    y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

    
def create_activation_datasets(activations_tensor, labels_tensor, val_size=0.1, test_size=0.2, balance=True, verbose=False):
    """
    Create datasets from pre-aggregated activations.

    Args:
        activations_tensor (tensor): tensor of pre-aggregated activations of shape [batch_size, dim] 
        labels_tensor (tensor): tensor of ground truth labels of shape [batch_size]
        verbose (bool): Should the function output to console. 
    """
    if balance:
        # Get indices for each label and subsample both classes to same size
        label_0_indices = (labels_tensor == 0.0).nonzero(as_tuple=True)[0]
        label_1_indices = (labels_tensor == 1.0).nonzero(as_tuple=True)[0]
        min_class_count = min(len(label_0_indices), len(label_1_indices))
        label_0_indices = label_0_indices[:min_class_count]
        label_1_indices = label_1_indices[:min_class_count]        

        # Compute split sizes
        n_total = min_class_count
        n_val = int(val_size * n_total)
        n_test = int(test_size * n_total)
        n_train = min_class_count - n_val - n_test

        # Split label 0s
        train_0 = label_0_indices[:n_train]
        val_0 = label_0_indices[n_train:n_train + n_val]
        test_0 = label_0_indices[n_train + n_val:]

        # Split label 1s
        train_1 = label_1_indices[:n_train]
        val_1 = label_1_indices[n_train:n_train + n_val]
        test_1 = label_1_indices[n_train + n_val:]

        # Concatenate splits and shuffle within each
        def get_split(indices_0, indices_1):
            indices = torch.cat([indices_0, indices_1])
            indices = indices[torch.randperm(len(indices))]
            return activations_tensor[indices], labels_tensor[indices]

        X_train, y_train = get_split(train_0, train_1)
        X_val, y_val = get_split(val_0, val_1)
        X_test, y_test = get_split(test_0, test_1)
    
    else:
        # Shuffle and split without balancing
        X_train, X_val, X_test, y_train, y_val, y_test = _train_val_test_split_torch(activations_tensor, labels_tensor, val_size, test_size)


    # Output balance
    if verbose:
        print(f"Train: {y_train.shape[0]} samples, {y_train.sum()} positives")
        print(f"Val:   {y_val.shape[0]} samples, {y_val.sum()} positives")
        print(f"Test:  {y_test.shape[0]} samples, {y_test.sum()} positives")
    
    train_dataset = {'X': X_train, 'y': y_train}
    val_dataset = {'X': X_val, 'y': y_val}
    if val_size == 0.0:
        val_dataset = None
    test_dataset = {'X': X_test, 'y': y_test}

    return train_dataset, val_dataset, test_dataset


    