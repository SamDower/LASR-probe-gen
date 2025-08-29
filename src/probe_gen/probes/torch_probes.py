import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from .base import Probe


class TorchLinearProbe(Probe):
    def __init__(self, cfg):
        """
        Initialize the PyTorch probe.
        Args:
            cfg (ConfigDict): ConfigDict with the following attributes:
                - use_bias (bool): Whether to use bias in the linear model.
                - normalize (bool): Whether to normalize input features.
                - lr (float): Learning rate for the optimizer.
                - weight_decay (float): L2 regularization parameter.
                - seed (int): Random seed.
        """
        super().__init__(cfg)
        
        # Set random seed
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Save the normalizing transformation parameters
        self.transformation_mean = None
        self.transformation_std = None
        
    def _build_model(self, input_dim):
        """Build the linear model"""
        self.model = nn.Linear(input_dim, 1, bias=self.cfg.use_bias)
        self.model.to(self.device)
        
    def _normalize_data(self, X, fit_transform=False):
        """Normalize the input data"""
        if not self.cfg.normalize:
            return X
            
        if fit_transform:
            self.transformation_mean = torch.mean(X, dim=0, keepdim=True)
            self.transformation_std = torch.std(X, dim=0, keepdim=True)
            # Avoid division by zero
            self.transformation_std = torch.where(
                self.transformation_std == 0, 
                torch.ones_like(self.transformation_std), 
                self.transformation_std
            )
        
        return (X - self.transformation_mean) / self.transformation_std
    
    def fit(self, train_dataset: dict, validation_dataset: dict = None) -> None:
        """
        Fits the probe to training data.
        Args:
            train_dataset (dict): train_dataset['X'] has shape [batch_size, dim], 
                                train_dataset['y'] has shape [batch_size].
            validation_dataset (dict, optional): validation_dataset['X'] has shape [batch_size, dim], 
                                               validation_dataset['y'] has shape [batch_size].
        """
        
        # Convert to tensors and move to device
        X_train = train_dataset['X'].to(self.device)
        y_train = train_dataset['y'].float().to(self.device)
        
        # Normalize data
        X_train = self._normalize_data(X_train, fit_transform=True)
        
        # Build model
        self._build_model(X_train.shape[1])
        
        # Setup optimizer
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.cfg.lr, 
            weight_decay=self.cfg.weight_decay
        )
        
        # Create data loader
        train_loader = DataLoader(
            TensorDataset(X_train, y_train), 
            batch_size=128, #self.cfg.batch_size
            shuffle=True
        )
        
        # Validation setup
        val_loader = None
        if validation_dataset is not None:
            X_val = validation_dataset['X'].to(self.device)
            y_val = validation_dataset['y'].float().to(self.device)
            X_val = self._normalize_data(X_val)
            val_loader = DataLoader(
                TensorDataset(X_val, y_val), 
                batch_size=128, #self.cfg.batch_size
                shuffle=False
            )
        
        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        print()
        self.model.train()
        num_epochs = 100  # self.cfg.epochs
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X).squeeze()
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X).squeeze()
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                self.model.train()
                
                # Early stopping check
                # if hasattr(self.cfg, 'early_stopping_patience'):
                if True:
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        best_model_state = self.model.state_dict().copy()
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= 10: # self.cfg.early_stopping_patience
                        # print(f"Early stopping at epoch {epoch+1}")
                        break
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, "
                          f"Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, "
                          f"Train Loss: {avg_train_loss:.4f}")
        
        # Load best model if early stopping was used
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        self.model.eval()
    
    def predict(self, X):
        """
        Get prediction labels (0 or 1) for the dataset.
        Args:
            X (tensor): tensor of aggregated activations with shape [batch_size, dim].
        Returns:
            y_pred (tensor): predicted labels of shape [batch_size].
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        self.model.eval()
        with torch.no_grad():
            X_input = X.to(self.device)
            X_normalized = self._normalize_data(X_input)
            logits = self.model(X_normalized).squeeze()
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float()
            
        return predictions.cpu()
    
    def predict_proba(self, X):
        """
        Get prediction probabilities of each point being class 1 for the dataset.
        Args:
            X (tensor): tensor of aggregated activations with shape [batch_size, dim].
        Returns:
            y_pred_proba (tensor): predicted probabilities of shape [batch_size].
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        self.model.eval()
        with torch.no_grad():
            X_input = X.to(self.device)
            X_normalized = self._normalize_data(X_input)
            logits = self.model(X_normalized).squeeze()
            probabilities = torch.sigmoid(logits)
            
        return probabilities.cpu()