import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
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
        
        X.sub_(self.transformation_mean.to(X.device)).div_(self.transformation_std.to(X.device))
        return X
    
    def fit(self, train_dataset: dict, validation_dataset: dict = None, verbose: bool = False) -> None:
        """
        Fits the probe to training data.
        Args:
            train_dataset (dict): train_dataset['X'] has shape [batch_size, dim], 
                                train_dataset['y'] has shape [batch_size].
            validation_dataset (dict, optional): validation_dataset['X'] has shape [batch_size, dim], 
                                               validation_dataset['y'] has shape [batch_size].
            verbose (bool, optional): Whether to print progress.
        """
        # TODO: copy stuff like num workers and blocking device from attention probe implementation
        with torch.no_grad():
            # Convert to tensors
            X_train = train_dataset['X']
            y_train = train_dataset['y'].float()
            
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
                X_val = validation_dataset['X']
                y_val = validation_dataset['y'].float()
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
            if verbose:
                print()
            self.model.train()
            
        num_epochs = 100  # self.cfg.epochs
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X.to(self.device)).squeeze()
                loss = self.criterion(outputs, batch_y.to(self.device))
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches
            
            # Validation
            with torch.no_grad():
                if val_loader is not None:
                    self.model.eval()
                    val_loss = 0.0
                    val_batches = 0
                    
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X.to(self.device)).squeeze()
                        loss = self.criterion(outputs, batch_y.to(self.device))
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
                    
                    if (epoch + 1) % 10 == 0 and verbose:
                        print(f"Epoch {epoch+1}/{num_epochs}, "
                            f"Train Loss: {avg_train_loss:.4f}, "
                            f"Val Loss: {avg_val_loss:.4f}")
                else:
                    if (epoch + 1) % 10 == 0 and verbose:
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
            
        return predictions
    
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
            
        return probabilities

    def eval(self, test_dataset):
        """
        Evaluates the probe on the test dataset using GPU-native operations.
        Args:
            test_dataset (dict): test_dataset['X'] has shape [batch_size, dim], test_dataset['y'] has shape [batch_size].
        Returns:
            results (dict): dictionary with the following keys:
                - accuracy (float): accuracy score.
                - roc_auc (float): roc auc score.
                - tpr_at_1_fpr (float): tpr at 1% fpr.
        """
        y = test_dataset['y']
        y_pred = self.predict(test_dataset['X'])
        y_pred_proba = self.predict_proba(test_dataset['X'])

        # accuracy = torch.mean((y_pred == y).float()).item()
        roc_auc = self._compute_roc_auc_gpu(y, y_pred_proba).item()
        tpr_at_1_fpr = None
        return {
            "roc_auc": roc_auc,
        }, y_pred, y_pred_proba

    def _compute_roc_auc_gpu(self, y_true, y_scores):
        """
        Compute ROC AUC using GPU-native operations.
        This implements the AUC calculation using the Wilcoxon-Mann-Whitney statistic.
        """
        # Ensure tensors are the right type
        y_true = y_true.float()
        y_scores = y_scores.float()
        
        # Get indices of positive and negative samples
        pos_mask = (y_true == 1)
        neg_mask = (y_true == 0)
        
        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()
        
        if n_pos == 0 or n_neg == 0:
            return torch.tensor(0.5)  # Undefined case, return 0.5
        
        # Get scores for positive and negative samples
        pos_scores = y_scores[pos_mask]
        neg_scores = y_scores[neg_mask]
        
        # Use broadcasting to compare all positive scores with all negative scores
        # pos_scores[:, None] creates shape [n_pos, 1]
        # neg_scores[None, :] creates shape [1, n_neg]
        # Broadcasting creates [n_pos, n_neg] comparison matrix
        comparisons = pos_scores[:, None] > neg_scores[None, :]
        ties = pos_scores[:, None] == neg_scores[None, :]
        
        # AUC = (number of correct comparisons + 0.5 * number of ties) / total comparisons
        auc = (comparisons.float().sum() + 0.5 * ties.float().sum()) / (n_pos * n_neg)
        
        return auc
    

class TorchAttentionProbe(Probe):
    def __init__(self, cfg):
        """
        Initialize the PyTorch attention probe.
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
        self.theta_q = None  # Query projection
        self.theta_v = None  # Value projection
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Save the normalizing transformation parameters
        self.transformation_mean = None
        self.transformation_std = None
        
    def _build_model(self, input_dim):
        """Build the linear models"""
        self.theta_q = nn.Linear(input_dim, 1, bias=self.cfg.use_bias)
        self.theta_v = nn.Linear(input_dim, 1, bias=self.cfg.use_bias)
        self.theta_q.to(self.device)
        self.theta_v.to(self.device)
        
    def _normalize_data(self, X, fit_transform=False):
        """Normalize the input data"""
        if not self.cfg.normalize:
            return X
            
        if fit_transform:
            # Compute mean and std across both batch and sequence dimensions
            self.transformation_mean = torch.mean(X, dim=(0, 1), keepdim=True)
            self.transformation_std = torch.std(X, dim=(0, 1), keepdim=True)
            # Avoid division by zero
            self.transformation_std = torch.where(
                self.transformation_std == 0, 
                torch.ones_like(self.transformation_std), 
                self.transformation_std
            )

        X.sub_(self.transformation_mean.to(X.device)).div_(self.transformation_std.to(X.device))
        return X
    
    def _compute_probe_output(self, X):
        """
        Compute the attention probe output: softmax(Aθ_q)ᵀ Aθ_v
        Args:
            X: Input activations of shape [batch_size, seq_len, activation_dim]
        Returns:
            output: Scalar output for each sample [batch_size, 1]
        """
        A_theta_q = self.theta_q(X).squeeze(-1)  # [batch_size, seq_len]
        A_theta_v = self.theta_v(X).squeeze(-1)  # [batch_size, seq_len]
        attention_weights = torch.softmax(A_theta_q, dim=1)  # [batch_size, seq_len]
        output = torch.sum(attention_weights * A_theta_v, dim=1, keepdim=True)  # [batch_size, 1]
        return output
    
    def fit(self, train_dataset: dict, validation_dataset: dict = None, verbose: bool = False) -> None:
        """
        Fits the probe to training data.
        Args:
            train_dataset (dict): train_dataset['X'] has shape [batch_size, seq_len, dim], 
                                train_dataset['y'] has shape [batch_size].
            validation_dataset (dict, optional): validation_dataset['X'] has shape [batch_size, seq_len, dim], 
                                               validation_dataset['y'] has shape [batch_size].
            verbose (bool, optional): Whether to print progress.
        """
        with torch.no_grad():
            # Convert to tensors and move to device
            X_train = train_dataset['X'].contiguous().float()
            y_train = train_dataset['y'].contiguous().float()
            
            # Normalize data
            X_train = self._normalize_data(X_train, fit_transform=True)
            
            # Build model
            self._build_model(X_train.shape[2])  # activation_dim is the last dimension
            
            # Setup optimizer
            parameters = list(self.theta_q.parameters()) + list(self.theta_v.parameters())
            optimizer = optim.Adam(
                parameters, 
                lr=self.cfg.lr, 
                weight_decay=self.cfg.weight_decay
            )
            
            # Create data loader with small batch size for gradient accumulation
            actual_batch_size = 32  # Small batch size that fits in GPU memory
            effective_batch_size = 128  # Effective batch size through gradient accumulation
            accumulation_steps = effective_batch_size // actual_batch_size
            
            num_workers = 16
            train_loader = DataLoader(
                TensorDataset(X_train, y_train),
                batch_size=actual_batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False,
                drop_last=False
            )
            
            # Validation setup
            val_loader = None
            if validation_dataset is not None:
                X_val = validation_dataset['X'].contiguous().float()
                y_val = validation_dataset['y'].contiguous().float()
                X_val = self._normalize_data(X_val)
                num_workers = 8
                val_loader = DataLoader(
                    TensorDataset(X_val, y_val),
                    batch_size=actual_batch_size*2,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=num_workers,
                    persistent_workers=True if num_workers > 0 else False,
                    drop_last=False
                )
            
            # Early stopping setup
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None
            
            # Training loop
            if verbose:
                print()
            self.theta_q.train()
            self.theta_v.train()
            
        num_epochs = 100
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for step, (batch_X, batch_y) in enumerate(train_loader):
                # Forward pass
                outputs = self._compute_probe_output(batch_X.to(self.device, non_blocking=True)).squeeze(-1)  # Only remove last dim, keep batch dim
                loss = self.criterion(outputs, batch_y.to(self.device, non_blocking=True))
                
                # Scale loss by accumulation steps to maintain equivalent gradients
                loss = loss / accumulation_steps
                
                # Backward pass (accumulate gradients)
                loss.backward()
                
                # Update weights every accumulation_steps
                if (step + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * accumulation_steps  # Scale back for logging
                num_batches += 1
            
            # Handle any remaining gradients at the end of epoch
            if len(train_loader) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
                
            avg_train_loss = epoch_loss / num_batches
            
            # Validation
            with torch.no_grad():
                if val_loader is not None:
                    self.theta_q.eval()
                    self.theta_v.eval()
                        
                    val_loss = 0.0
                    val_batches = 0
                    
                    for batch_X, batch_y in val_loader:
                        outputs = self._compute_probe_output(batch_X.to(self.device, non_blocking=True)).squeeze(-1)  # Only remove last dim, keep batch dim
                        loss = self.criterion(outputs, batch_y.to(self.device, non_blocking=True))
                        val_loss += loss.item()
                        val_batches += 1
                    
                    # Sometimes is 0 for some reason
                    if val_batches != 0:
                        avg_val_loss = val_loss / val_batches
                        self.theta_q.train()
                        self.theta_v.train()
                        
                        # Early stopping check
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            patience_counter = 0
                            best_model_state = {
                                'theta_q': self.theta_q.state_dict().copy(),
                                'theta_v': self.theta_v.state_dict().copy()
                            }
                        else:
                            patience_counter += 1
                        
                    if patience_counter >= 10: # self.cfg.early_stopping_patience
                        # print(f"Early stopping at epoch {epoch+1}")
                        break
                    
                    if (epoch + 1) % 10 == 0 and verbose:
                        print(f"Epoch {epoch+1}/{num_epochs}, "
                            f"Train Loss: {avg_train_loss:.4f}, "
                            f"Val Loss: {avg_val_loss:.4f}")
                else:
                    if (epoch + 1) % 10 == 0 and verbose:
                        print(f"Epoch {epoch+1}/{num_epochs}, "
                            f"Train Loss: {avg_train_loss:.4f}")
        
        # Load best model if early stopping was used
        if best_model_state is not None:
            self.theta_q.load_state_dict(best_model_state['theta_q'])
            self.theta_v.load_state_dict(best_model_state['theta_v'])
        
        # Set to eval mode
        self.theta_q.eval()
        self.theta_v.eval()
    
    def predict(self, X):
        """
        Get prediction labels (0 or 1) for the dataset.
        Args:
            X (tensor): tensor of activations with shape [batch_size, seq_len, dim].
        Returns:
            y_pred (tensor): predicted labels of shape [batch_size].
        """
        if self.theta_q is None or self.theta_v is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        self.theta_q.eval()
        self.theta_v.eval()
            
        with torch.no_grad():
            X_input = X.to(self.device)
            X_normalized = self._normalize_data(X_input)
            logits = self._compute_probe_output(X_normalized).squeeze(-1)  # Only remove last dim, keep batch dim
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float()
            
        return predictions
    
    def predict_proba(self, X):
        """
        Get prediction probabilities of each point being class 1 for the dataset.
        Args:
            X (tensor): tensor of activations with shape [batch_size, seq_len, dim].
        Returns:
            y_pred_proba (tensor): predicted probabilities of shape [batch_size].
        """
        if self.theta_q is None or self.theta_v is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        self.theta_q.eval()
        self.theta_v.eval()
            
        with torch.no_grad():
            X_input = X.to(self.device)
            X_normalized = self._normalize_data(X_input)
            logits = self._compute_probe_output(X_normalized).squeeze(-1)  # Only remove last dim, keep batch dim
            probabilities = torch.sigmoid(logits)
            
        return probabilities

    def eval(self, test_dataset):
        """
        Evaluates the probe on the test dataset using GPU-native operations.
        Args:
            test_dataset (dict): test_dataset['X'] has shape [batch_size, dim], test_dataset['y'] has shape [batch_size].
        Returns:
            results (dict): dictionary with the following keys:
                - accuracy (float): accuracy score.
                - roc_auc (float): roc auc score.
                - tpr_at_1_fpr (float): tpr at 1% fpr.
        """
        y = test_dataset['y']
        y_pred = self.predict(test_dataset['X'])
        y_pred_proba = self.predict_proba(test_dataset['X'])

        # accuracy = torch.mean((y_pred == y).float()).item()
        roc_auc = self._compute_roc_auc_gpu(y, y_pred_proba).item()
        tpr_at_1_fpr = None
        return {
            "roc_auc": roc_auc,
        }, y_pred, y_pred_proba

    def _compute_roc_auc_gpu(self, y_true, y_scores):
        """
        Compute ROC AUC using GPU-native operations.
        This implements the AUC calculation using the Wilcoxon-Mann-Whitney statistic.
        """
        # Ensure tensors are the right type
        y_true = y_true.float()
        y_scores = y_scores.float()
        
        # Get indices of positive and negative samples
        pos_mask = (y_true == 1)
        neg_mask = (y_true == 0)
        
        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()
        
        if n_pos == 0 or n_neg == 0:
            return torch.tensor(0.5)  # Undefined case, return 0.5
        
        # Get scores for positive and negative samples
        pos_scores = y_scores[pos_mask]
        neg_scores = y_scores[neg_mask]
        
        # Use broadcasting to compare all positive scores with all negative scores
        # pos_scores[:, None] creates shape [n_pos, 1]
        # neg_scores[None, :] creates shape [1, n_neg]
        # Broadcasting creates [n_pos, n_neg] comparison matrix
        comparisons = pos_scores[:, None] > neg_scores[None, :]
        ties = pos_scores[:, None] == neg_scores[None, :]
        
        # AUC = (number of correct comparisons + 0.5 * number of ties) / total comparisons
        auc = (comparisons.float().sum() + 0.5 * ties.float().sum()) / (n_pos * n_neg)
        
        return auc
