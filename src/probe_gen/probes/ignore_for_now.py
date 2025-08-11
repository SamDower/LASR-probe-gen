from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from probe_gen.get_activations import ActivationDataset


class LinearProbe(nn.Module):
    """Linear probe for binary classification of activations"""

    def __init__(self, hidden_dim: int, probe_type: str = "mean"):
        """
        Args:
            hidden_dim: Dimension of the activation vectors (D)
            probe_type: Type of probe aggregation ("mean", "max", "last", "attention", "softmax")
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.probe_type = probe_type

        if probe_type == "attention":
            # Attention probe uses query and value vectors
            self.theta_q = nn.Parameter(torch.randn(hidden_dim))
            self.theta_v = nn.Parameter(torch.randn(hidden_dim))
        elif probe_type == "softmax":
            # Softmax probe with temperature
            self.theta = nn.Parameter(torch.randn(hidden_dim))
            self.temperature = 5.0  # Based on paper
        else:
            # Simple linear probe
            self.theta = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the probe

        Args:
            activations: Tensor of shape (batch_size, seq_len, hidden_dim)

        Returns:
            logits: Tensor of shape (batch_size,)
        """
        if self.probe_type == "mean":
            # Mean aggregation: average across sequence dimension
            mean_activations = torch.mean(
                activations, dim=1
            )  # (batch_size, hidden_dim)
            logits = torch.matmul(mean_activations, self.theta)

        elif self.probe_type == "max":
            # Max aggregation: take max score across sequence positions
            scores = torch.matmul(activations, self.theta)  # (batch_size, seq_len)
            logits = torch.max(scores, dim=1)[0]  # (batch_size,)

        elif self.probe_type == "last":
            # Last token aggregation
            last_activations = activations[:, -1, :]  # (batch_size, hidden_dim)
            logits = torch.matmul(last_activations, self.theta)

        elif self.probe_type == "attention":
            # Attention-based aggregation
            queries = torch.matmul(activations, self.theta_q)  # (batch_size, seq_len)
            values = torch.matmul(activations, self.theta_v)  # (batch_size, seq_len)

            # Apply softmax to queries to get attention weights
            attention_weights = torch.softmax(queries, dim=1)  # (batch_size, seq_len)

            # Weighted sum of values
            logits = torch.sum(attention_weights * values, dim=1)  # (batch_size,)

        elif self.probe_type == "softmax":
            # Softmax aggregation with temperature
            scores = (
                torch.matmul(activations, self.theta) / self.temperature
            )  # (batch_size, seq_len)
            weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len)

            # Weighted sum of original scores
            original_scores = torch.matmul(
                activations, self.theta
            )  # (batch_size, seq_len)
            logits = torch.sum(weights * original_scores, dim=1)  # (batch_size,)

        else:
            raise ValueError(f"Unknown probe type: {self.probe_type}")

        return logits


def train_probe(
    train_activations: torch.Tensor,
    train_labels: torch.Tensor,
    val_activations: torch.Tensor = None,
    val_labels: torch.Tensor = None,
    probe_type: str = "mean",
    learning_rate: float = 5e-3,
    weight_decay: float = 1e-3,
    batch_size: int = 16,
    num_epochs: int = 2,
    early_stop_patience: int = 50,
    device: str = "cpu",
    silent: bool = False,
) -> Tuple[LinearProbe, List[float], List[float]]:
    """
    Train a linear probe on activation data

    Args:
        train_activations: Training activations of shape (N, S, D)
        train_labels: Training labels of shape (N,)
        val_activations: Validation activations (optional)
        val_labels: Validation labels (optional)
        probe_type: Type of probe to train
        learning_rate: Learning rate for optimization
        weight_decay: Weight decay for regularization
        batch_size: Batch size for training
        num_epochs: Maximum number of epochs
        early_stop_patience: Patience for early stopping
        device: Device to train on ("cpu" or "cuda")

    Returns:
        Tuple of (trained_model, train_losses, val_losses)
    """

    # Move data to device
    device = torch.device(device)
    train_activations = train_activations.to(device)
    train_labels = train_labels.to(device)

    if val_activations is not None:
        val_activations = val_activations.to(device)
        val_labels = val_labels.to(device)

    # Create datasets and dataloaders
    train_dataset = ActivationDataset(train_activations, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if val_activations is not None:
        val_dataset = ActivationDataset(val_activations, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    hidden_dim = train_activations.shape[-1]
    model = LinearProbe(hidden_dim, probe_type).to(device)

    # Initialize optimizer and loss function
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    # Learning rate scheduler (cosine annealing as suggested in paper)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate / 10
    )

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        for batch_activations, batch_labels in train_loader:
            optimizer.zero_grad()

            logits = model(batch_activations)
            loss = criterion(logits, batch_labels.float())

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        if val_activations is not None:
            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for batch_activations, batch_labels in val_loader:
                    logits = model(batch_activations)
                    loss = criterion(logits, batch_labels.float())
                    epoch_val_loss += loss.item()

            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if not silent:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
                )
        elif not silent:
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        scheduler.step()

    # Load best model if validation was used
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses


def evaluate_probe(
    model: LinearProbe,
    test_activations: torch.Tensor,
    test_labels: torch.Tensor,
    device: str = "cpu",
) -> dict:
    """
    Evaluate a trained probe on test data

    Args:
        model: Trained LinearProbe model
        test_activations: Test activations of shape (N, S, D)
        test_labels: Test labels of shape (N,)
        device: Device to run evaluation on

    Returns:
        Dictionary with evaluation metrics
    """
    device = torch.device(device)
    model = model.to(device)
    test_activations = test_activations.to(device)
    test_labels = test_labels.to(device)

    model.eval()

    with torch.no_grad():
        logits = model(test_activations)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).float()

    # Convert to numpy for sklearn metrics
    test_labels_np = test_labels.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    probabilities_np = probabilities.cpu().numpy()

    # Calculate metrics
    accuracy = accuracy_score(test_labels_np, predictions_np)
    auc_roc = roc_auc_score(test_labels_np, probabilities_np)

    # Calculate TPR at 1% FPR (as mentioned in paper)

    fpr, tpr, thresholds = roc_curve(test_labels_np, probabilities_np)

    # Find TPR at 1% FPR
    target_fpr = 0.01
    idx = np.argmax(fpr >= target_fpr)
    tpr_at_1_fpr = tpr[idx] if idx < len(tpr) else 0

    return {
        "accuracy": accuracy,
        "auc_roc": auc_roc,
        "tpr_at_1_fpr": tpr_at_1_fpr,
        "predictions": predictions_np,
        "probabilities": probabilities_np,
    }
