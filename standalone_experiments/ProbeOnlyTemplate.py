"""
Standalone Probe Experiment Script

Complete pipeline for probe training and evaluation without dependencies on probe_gen.
Loads activations and labels directly from HuggingFace.
"""

import json
import os
import shutil
import tempfile
from typing import Dict, List, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# Configuration
# ============================================================================

HF_REPO_ID_TEMPLATE = "lasrprobegen/{behaviour}-activations"

# Training dataset configuration
# Check behaviours and datasources and models in src/probe_gen/config.py
BEHAVIOUR = "sycophancy" 
DATASOURCE = "multichoice" 
RESPONSE_MODEL = "llama_3b"
ACTIVATIONS_MODEL = "llama_3b" 
LAYER = 12 # (every 3rd layer is on HF)

# Test dataset configuration
# Set to any value to None to use same as training, or specify different values
TEST_BEHAVIOUR = None
TEST_DATASOURCE = "arguments" # "arguments" for OOD, None for ID
TEST_RESPONSE_MODEL = None

# Dataset splits
SPLIT_VAL = 3500 
SPLIT_VAL_SIZE = 500
TEST_SIZE = 1000

# Probe hyperparameters
PROBE_TRAINING_METHOD = "sklearn"  # "adam" or "sklearn"
PROBE_LR = 0.001
PROBE_WEIGHT_DECAY = 0.01
PROBE_NORMALIZE = True
PROBE_USE_BIAS = True
PROBE_SEED = 42
PROBE_EPOCHS = 100
PROBE_PATIENCE = 10
PROBE_C = 1.0  # For sklearn logistic regression (inverse of regularization strength)

# File I/O configuration
OUTPUT_DIR = "standalone_experiments/experiment_data"  # Directory to save intermediate results


# ============================================================================
# Load Activations and Labels from HuggingFace
# ============================================================================

def _download_labels_from_hf(repo_id: str, labels_filepath: str, output_path: str):
    """Download labels file from HuggingFace to a specific output path (flattened, no subdirectories)."""
    # Download to temp directory first (preserves HF directory structure)
    with tempfile.TemporaryDirectory() as tmpdir:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=labels_filepath,
            local_dir=tmpdir,
            token=os.environ.get("HF_TOKEN")
        )
        # Move the file to the desired output path (flattened)
        shutil.move(downloaded_path, output_path)


def _load_labels_from_local_jsonl(labels_filename: str, verbose: bool = False) -> tuple[torch.Tensor, List[Dict]]:
    """Load labels and full data from local JSONL file."""
    labels_list = []
    data_rows = []
    with open(labels_filename, 'r') as file:
        for line in file:
            data = json.loads(line)
            data_rows.append(data)
            # Use 'labels' field directly (should be "positive" or "negative")
            if data.get("labels") == "positive":
                labels_list.append(1.0)
            elif data.get("labels") == "negative":
                labels_list.append(0.0)
            else:
                # Fallback to scale_labels thresholding if labels field not present
                if data.get("scale_labels", 5) <= 5:
                    labels_list.append(1.0)
                else:
                    labels_list.append(0.0)
    labels_tensor = torch.tensor(labels_list)
    if verbose:
        print("Loaded labels")
    return labels_tensor, data_rows


def _load_activations_from_hf(repo_id: str, filename: str, verbose: bool = False):
    """Load activations from HuggingFace dataset."""
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
        print(f"Loaded activations with shape {activations_tensor.shape}")

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
        print(f"Calculated attention mask with shape {attention_mask.shape}")

    return activations_tensor, attention_mask


def load_hf_activations_at_layer(
    behaviour: str,
    datasource: str,
    activations_model: str = "llama_3b",
    response_model: str = "llama_3b",
    generation_method: str = "on_policy",
    mode: str = "train",
    layer: int = 12,
    and_labels: bool = False,
    verbose: bool = False
):
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

    repo_id = HF_REPO_ID_TEMPLATE.format(behaviour=behaviour)
    filepath = f"{datasource}/{activations_model}/{response_model}_{generation_method}_{mode}_layer_{layer}.pkl"

    try:
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
        # Use just the filename, not the full path structure (flattened)
        labels_filename = f"{datasource}_{response_model}_{generation_method_for_labels}_{mode}.jsonl"
        labels_fullpath = os.path.join(OUTPUT_DIR, labels_filename)
        
        # For downloading, we still need the original path structure from HF
        labels_filepath_hf = f"{datasource}/{response_model}_{generation_method_for_labels}_{mode}.jsonl"

        try:
            if not os.path.exists(labels_fullpath):
                _download_labels_from_hf(repo_id, labels_filepath_hf, labels_fullpath)
            labels_tensor, data_rows = _load_labels_from_local_jsonl(labels_fullpath, verbose)
        except Exception as e:
            # Try loading with on_policy in the name instead
            if generation_method == "off_policy":
                real_generation_method = "incentivised" if behaviour in ["deception", "sandbagging"] else "on_policy"
                labels_filename = labels_filename.replace("off_policy", real_generation_method)
                labels_filepath_hf = labels_filepath_hf.replace("off_policy", real_generation_method)
                labels_fullpath = os.path.join(OUTPUT_DIR, labels_filename)
                if not os.path.exists(labels_fullpath):
                    _download_labels_from_hf(repo_id, labels_filepath_hf, labels_fullpath)
                labels_tensor, data_rows = _load_labels_from_local_jsonl(labels_fullpath, verbose)
            else:
                raise e
        return activations_tensor, attention_mask, labels_tensor, data_rows

    else:
        return activations_tensor, attention_mask, None


# ============================================================================
# Create Activation Datasets (Split into train/val/test)
# ============================================================================

def mean_pool_activations(activations: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool activations across sequence length, masking padding tokens."""
    mask = attention_mask.unsqueeze(-1).float()
    masked_activations = activations * mask
    pooled = masked_activations.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled


def create_activation_datasets(activations_tensor, labels_tensor, data_rows: List[Dict], splits=[3500, 500, 0], verbose=False):
    """
    Create datasets from pre-aggregated activations.

    Args:
        activations_tensor (tensor): tensor of pre-aggregated activations of shape [batch_size, dim]
        labels_tensor (tensor): tensor of ground truth labels of shape [batch_size]
        data_rows (list): List of dicts with full data from JSONL (input, output, labels, scale_labels, etc.)
        splits (list): [train_size, val_size, test_size]
        verbose (bool): Should the function output to console.

    Returns:
        splits_dict (dict): {'train': [{'input': ..., 'output': ..., 'scale_labels': ..., 'label': ...}, ...], ...}
        activations_dict (dict): {'train': activations_tensor, 'val': activations_tensor, 'test': activations_tensor}
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
        return indices

    train_indices = get_split(train_0, train_1)
    val_indices = get_split(val_0, val_1)
    test_indices = get_split(test_0, test_1)

    # Create splits dict with only essential fields
    def create_split_data(indices):
        split_data = []
        for idx in indices:
            idx_int = idx.item()
            row = data_rows[idx_int]
            
            # Extract input and output from inputs field
            input_content = None
            output_content = None
            
            if 'inputs' in row:
                # inputs might be a JSON string or a list
                inputs_data = row['inputs']
                if isinstance(inputs_data, str):
                    inputs_data = json.loads(inputs_data)
                
                if isinstance(inputs_data, list) and len(inputs_data) >= 2:
                    input_content = inputs_data[0].get('content', '')
                    output_content = inputs_data[1].get('content', '')
            
            # Fallback to existing fields if inputs parsing failed
            if input_content is None:
                input_content = row.get('input', row.get('input_formatted', ''))
            if output_content is None:
                output_content = row.get('output', row.get('model_outputs', ''))
            
            # Extract label
            label = row.get('labels', '')
            if not label:
                # Fallback: derive from scale_labels
                if row.get('scale_labels', 5) <= 5:
                    label = 'positive'
                else:
                    label = 'negative'
            
            # Create simplified data entry with only required fields
            simplified_row = {
                'input': input_content,
                'output': output_content,
                'scale_labels': row.get('scale_labels'),
                'label': label
            }
            split_data.append(simplified_row)
        return split_data
    
    splits_dict = {
        'train': create_split_data(train_indices),
        'val': create_split_data(val_indices),
        'test': create_split_data(test_indices),
    }

    # Create activations dict
    activations_dict = {
        'train': activations_tensor[train_indices],
        'val': activations_tensor[val_indices],
        'test': activations_tensor[test_indices],
    }

    # Output balance
    if verbose:
        train_labels = labels_tensor[train_indices]
        val_labels = labels_tensor[val_indices]
        test_labels = labels_tensor[test_indices]
        print(f"Train: {train_labels.shape[0]} samples, {train_labels.sum()} positives")
        print(f"Val:   {val_labels.shape[0]} samples, {val_labels.sum()} positives")
        print(f"Test:  {test_labels.shape[0]} samples, {test_labels.sum()} positives")

    return splits_dict, activations_dict


def save_splits(splits: Dict[str, List[Dict]], filename: str = "splits.json"):
    """Save splits to JSON file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Saved splits to {filepath}")


def load_splits(filename: str = "splits.json") -> Dict[str, List[Dict]]:
    """Load splits from JSON file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'r') as f:
        splits = json.load(f)
    print(f"Loaded splits from {filepath}")
    return splits


def save_activations(activations: Dict[str, torch.Tensor], filename: str = "activations.pt"):
    """Save activations to PyTorch file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    torch.save(activations, filepath)
    print(f"Saved activations to {filepath}")


def load_activations(filename: str = "activations.pt") -> Dict[str, torch.Tensor]:
    """Load activations from PyTorch file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    activations = torch.load(filepath)
    print(f"Loaded activations from {filepath}")
    return activations


# ============================================================================
# Create and Train Probe
# ============================================================================

class TorchLinearProbe(nn.Module):
    """PyTorch linear probe for binary classification with mean pooling aggregation."""
    
    def __init__(self, input_dim: int, normalize: bool = True, use_bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=use_bias)
        self.normalize = normalize
        self.mean = None
        self.std = None
    
    def fit_normalization(self, activations: torch.Tensor, attention_mask: torch.Tensor):
        """Fit normalization parameters on pooled activations."""
        if self.normalize:
            # Pool first, then compute stats
            pooled = mean_pool_activations(activations, attention_mask)
            self.mean = pooled.mean(dim=0, keepdim=True)
            self.std = pooled.std(dim=0, keepdim=True)
            self.std = torch.where(self.std == 0, torch.ones_like(self.std), self.std)
    
    def normalize_input(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize input."""
        if self.normalize and self.mean is not None:
            return (X - self.mean.to(X.device)) / self.std.to(X.device)
        return X
    
    def forward(self, activations: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass: pool, normalize, then linear layer."""
        # Pool activations: [batch, seq_len, hidden_dim] -> [batch, hidden_dim]
        pooled = mean_pool_activations(activations, attention_mask)
        # Normalize
        pooled_norm = self.normalize_input(pooled)
        # Linear layer
        return self.linear(pooled_norm).squeeze(-1)


def create_and_train_probe_adam(
    train_dataset: Dict[str, torch.Tensor],
    val_dataset: Optional[Dict[str, torch.Tensor]] = None,
) -> TorchLinearProbe:
    """Create and train a PyTorch linear probe using Adam optimizer."""
    train_X = train_dataset['X']
    train_y = train_dataset['y']
    
    # Check if activations are already pooled (2D) or need pooling (3D)
    if len(train_X.shape) == 2:
        # Already pooled: [batch, dim]
        input_dim = train_X.shape[1]
        use_pooling = False
    else:
        # Need pooling: [batch, seq_len, dim]
        input_dim = train_X.shape[2]
        use_pooling = True
        # Create dummy masks if not provided (all ones)
        train_masks = torch.ones(train_X.shape[0], train_X.shape[1])
    
    probe = TorchLinearProbe(input_dim, normalize=PROBE_NORMALIZE, use_bias=PROBE_USE_BIAS)
    print(f"Created TorchLinearProbe with input_dim={input_dim}")
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    probe = probe.to(device)
    train_X = train_X.to(device)
    train_y = train_y.float().to(device)
    
    # Fit normalization
    if use_pooling:
        train_masks = train_masks.to(device)
        probe.fit_normalization(train_X, train_masks)
    else:
        # For already pooled activations, fit on the pooled data directly
        if PROBE_NORMALIZE:
            probe.mean = train_X.mean(dim=0, keepdim=True)
            probe.std = train_X.std(dim=0, keepdim=True)
            probe.std = torch.where(probe.std == 0, torch.ones_like(probe.std), probe.std)
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(probe.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY)
    
    if use_pooling:
        train_tensor_dataset = TensorDataset(train_X, train_masks, train_y)
    else:
        train_tensor_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_tensor_dataset, batch_size=128, shuffle=True)
    
    # Setup validation if available
    val_loader = None
    if val_dataset is not None:
        val_X = val_dataset['X'].to(device)
        val_y = val_dataset['y'].float().to(device)
        if use_pooling:
            val_masks = torch.ones(val_X.shape[0], val_X.shape[1]).to(device)
            val_tensor_dataset = TensorDataset(val_X, val_masks, val_y)
        else:
            val_tensor_dataset = TensorDataset(val_X, val_y)
        val_loader = DataLoader(val_tensor_dataset, batch_size=128, shuffle=False)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(PROBE_EPOCHS):
        # Train
        probe.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            if use_pooling:
                acts_batch, masks_batch, y_batch = batch
                logits = probe(acts_batch, masks_batch)
            else:
                acts_batch, y_batch = batch
                # For already pooled, just pass through linear layer
                acts_norm = probe.normalize_input(acts_batch)
                logits = probe.linear(acts_norm).squeeze(-1)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        if val_loader is not None:
            probe.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if use_pooling:
                        acts_batch, masks_batch, y_batch = batch
                        logits = probe(acts_batch, masks_batch)
                    else:
                        acts_batch, y_batch = batch
                        acts_norm = probe.normalize_input(acts_batch)
                        logits = probe.linear(acts_norm).squeeze(-1)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = probe.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= PROBE_PATIENCE:
                    print(f"Early stopping at epoch {epoch}")
                    break
        else:
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}")
    
    # Load best model
    if best_state is not None:
        probe.load_state_dict(best_state)
    
    print("Training complete!")
    return probe


def create_and_train_probe_sklearn(
    train_dataset: Dict[str, torch.Tensor],
    val_dataset: Optional[Dict[str, torch.Tensor]] = None,
) -> LogisticRegression:
    """Create and train a scikit-learn logistic regression probe."""
    train_X = train_dataset['X']
    train_y = train_dataset['y']
    
    # Check if activations are already pooled (2D) or need pooling (3D)
    if len(train_X.shape) == 3:
        # Need pooling: [batch, seq_len, dim]
        print("Mean pooling activations...")
        # Create dummy masks if not provided (all ones)
        train_masks = torch.ones(train_X.shape[0], train_X.shape[1])
        train_X = mean_pool_activations(train_X, train_masks)
    
    train_X = train_X.cpu().numpy()
    train_y = train_y.cpu().numpy()
    
    if val_dataset is not None:
        val_X = val_dataset['X']
        val_y = val_dataset['y']
        if len(val_X.shape) == 3:
            val_masks = torch.ones(val_X.shape[0], val_X.shape[1])
            val_X = mean_pool_activations(val_X, val_masks)
        val_X = val_X.cpu().numpy()
        val_y = val_y.cpu().numpy()
    else:
        val_X = None
        val_y = None
    
    # Normalize if needed
    if PROBE_NORMALIZE:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        if val_X is not None:
            val_X = scaler.transform(val_X)
    else:
        scaler = None
    
    # Train logistic regression
    print("Fitting LogisticRegression...")
    clf = LogisticRegression(
        C=PROBE_C,
        fit_intercept=PROBE_USE_BIAS,
        max_iter=1000,
        random_state=PROBE_SEED,
        solver='lbfgs',
    )
    clf.fit(train_X, train_y)
    
    # Store scaler for later use
    clf.scaler = scaler
    
    # Print validation accuracy if available
    if val_X is not None:
        val_pred = clf.predict(val_X)
        val_acc = accuracy_score(val_y, val_pred)
        print(f"Validation accuracy: {val_acc:.4f}")
    
    print("Training complete!")
    return clf


# ============================================================================
# Evaluate Probe
# ============================================================================

def evaluate_probe(
    probe,
    test_dataset: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Evaluate the probe on test set."""
    test_X = test_dataset['X']
    test_y = test_dataset['y']
    
    # Determine probe type by checking if it's a sklearn LogisticRegression
    if isinstance(probe, LogisticRegression):
        # sklearn probe
        if len(test_X.shape) == 3:
            # Need pooling
            test_masks = torch.ones(test_X.shape[0], test_X.shape[1])
            test_X = mean_pool_activations(test_X, test_masks)
        
        test_X = test_X.cpu().numpy()
        
        if hasattr(probe, 'scaler') and probe.scaler is not None:
            test_X = probe.scaler.transform(test_X)
        
        probs = probe.predict_proba(test_X)[:, 1]
        preds = probe.predict(test_X)
        y_true = test_y.cpu().numpy()
    else:
        # PyTorch probe (or any other non-sklearn probe)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        probe.eval()
        
        with torch.no_grad():
            test_X = test_X.to(device)
            test_y = test_y.to(device)
            
            if len(test_X.shape) == 3:
                # Need pooling
                test_masks = torch.ones(test_X.shape[0], test_X.shape[1]).to(device)
                logits = probe(test_X, test_masks)
            else:
                # Already pooled
                test_X_norm = probe.normalize_input(test_X)
                logits = probe.linear(test_X_norm).squeeze(-1)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            y_true = test_y.cpu().numpy()
    y_proba = probs
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, preds)
    auroc = roc_auc_score(y_true, y_proba)
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    target_fpr = 0.01
    idx = np.argmax(fpr >= target_fpr)
    tpr_at_1_fpr = tpr[idx] if idx < len(tpr) else 0.0
    
    results = {
        'accuracy': accuracy,
        'auroc': auroc,
        'tpr_at_1_fpr': tpr_at_1_fpr,
    }
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    print("=" * 60)
    
    return results


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Run the complete pipeline."""
    print("\n" + "=" * 60)
    print("STANDALONE PROBE EXPERIMENT")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(PROBE_SEED)
    np.random.seed(PROBE_SEED)
    
    # Load training activations and labels from HuggingFace
    print("\n" + "=" * 60)
    print("Loading training activations and labels from HuggingFace")
    print("=" * 60)
    
    # Adjust split size based on behaviour/datasource (matching TrainProbe.ipynb logic)
    split_val = SPLIT_VAL
    if BEHAVIOUR in ["deception", "sandbagging"]:
        split_val = 2500
    if DATASOURCE == "shakespeare":
        split_val = 3000
    
    splits_file = "splits.json"
    activations_file = "activations.pt"
    splits_filepath = os.path.join(OUTPUT_DIR, splits_file)
    activations_filepath = os.path.join(OUTPUT_DIR, activations_file)
    
    if os.path.exists(splits_filepath) and os.path.exists(activations_filepath):
        print("Loading splits and activations from files...")
        splits = load_splits(splits_file)
        activations = load_activations(activations_file)
        
        # Extract labels from splits.json (derive binary from label field)
        train_labels = torch.tensor([1.0 if item['label'] == 'positive' else 0.0 for item in splits['train']], dtype=torch.float32)
        val_labels = torch.tensor([1.0 if item['label'] == 'positive' else 0.0 for item in splits['val']], dtype=torch.float32)
        
        # Create datasets
        train_dataset = {'X': activations['train'], 'y': train_labels}
        val_dataset = {'X': activations['val'], 'y': val_labels}
    else:
        # Load training activations and labels (mode="train")
        activations_tensor, attention_mask, labels_tensor, data_rows = load_hf_activations_at_layer(
            BEHAVIOUR,
            DATASOURCE,
            ACTIVATIONS_MODEL,
            RESPONSE_MODEL,
            "on_policy",  # Always use "on_policy" generation method
            "train",  # Always use "train" mode for training data
            LAYER,
            and_labels=True,
            verbose=True
        )
        
        # Apply mean pooling (before splitting)
        print("\nMean pooling activations...")
        activations_tensor = mean_pool_activations(activations_tensor, attention_mask)
        print(f"Pooled activations shape: {activations_tensor.shape}")
        
        # Create train/val splits (no test split - test is loaded separately)
        print("\n" + "=" * 60)
        print("Creating train/val splits")
        print("=" * 60)
        
        splits, activations = create_activation_datasets(
            activations_tensor,
            labels_tensor,
            data_rows,
            splits=[split_val, SPLIT_VAL_SIZE, 0],  # No test split
            verbose=True
        )
        
        # Save splits (JSON) and activations (PyTorch) separately
        save_splits(splits, splits_file)
        save_activations(activations, activations_file)
        
        # Extract labels from splits.json (derive binary from label field)
        train_labels = torch.tensor([1.0 if item['label'] == 'positive' else 0.0 for item in splits['train']], dtype=torch.float32)
        val_labels = torch.tensor([1.0 if item['label'] == 'positive' else 0.0 for item in splits['val']], dtype=torch.float32)
        
        # Create datasets
        train_dataset = {'X': activations['train'], 'y': train_labels}
        val_dataset = {'X': activations['val'], 'y': val_labels}
    
    # Create and train probe
    print("\n" + "=" * 60)
    print("Creating and training probe")
    print("=" * 60)
    
    if PROBE_TRAINING_METHOD == "adam":
        probe = create_and_train_probe_adam(
            train_dataset,
            val_dataset,
        )
    elif PROBE_TRAINING_METHOD == "sklearn":
        probe = create_and_train_probe_sklearn(
            train_dataset,
            val_dataset,
        )
    else:
        raise ValueError(f"Unknown training method: {PROBE_TRAINING_METHOD}")
    
    # Load separate test dataset from HuggingFace
    print("\n" + "=" * 60)
    print("Loading test dataset from HuggingFace")
    print("=" * 60)
    
    # Determine test set parameters (use training params if None)
    test_behaviour = TEST_BEHAVIOUR if TEST_BEHAVIOUR is not None else BEHAVIOUR
    test_datasource = TEST_DATASOURCE if TEST_DATASOURCE is not None else DATASOURCE
    # Always use same activations model and layer as training
    test_activations_model = ACTIVATIONS_MODEL
    test_response_model = TEST_RESPONSE_MODEL if TEST_RESPONSE_MODEL is not None else RESPONSE_MODEL
    test_layer = LAYER
    
    print(f"Test set configuration:")
    print(f"  Behaviour: {test_behaviour}")
    print(f"  Datasource: {test_datasource}")
    print(f"  Activations model: {test_activations_model}")
    print(f"  Response model: {test_response_model}")
    print(f"  Generation method: on_policy")
    print(f"  Layer: {test_layer}")
    print(f"  Test size: {TEST_SIZE}")
    
    # Load test activations and labels (mode="test", always "on_policy")
    test_activations_tensor, test_attention_mask, test_labels_tensor, test_data_rows = load_hf_activations_at_layer(
        test_behaviour,
        test_datasource,
        test_activations_model,
        test_response_model,
        "on_policy",  # Always use "on_policy" generation method
        "test",  # Always use "test" mode for test data
        test_layer,
        and_labels=True,
        verbose=True
    )
    
    # Apply mean pooling to test data
    print("\nMean pooling test activations...")
    test_activations_tensor = mean_pool_activations(test_activations_tensor, test_attention_mask)
    print(f"Pooled test activations shape: {test_activations_tensor.shape}")
    
    # Create test dataset (splits=[0, 0, test_size] means all data goes to test)
    test_splits, test_activations = create_activation_datasets(
        test_activations_tensor,
        test_labels_tensor,
        test_data_rows,
        splits=[0, 0, TEST_SIZE],
        verbose=True
    )
    
    # Extract labels from splits (derive binary from label field)
    test_labels = torch.tensor([1.0 if item['label'] == 'positive' else 0.0 for item in test_splits['test']], dtype=torch.float32)
    test_dataset = {'X': test_activations['test'], 'y': test_labels}
    
    # Evaluate probe on test set
    print("\n" + "=" * 60)
    print("Evaluating probe on test set")
    print("=" * 60)
    
    results = evaluate_probe(
        probe,
        test_dataset,
    )
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = main()

