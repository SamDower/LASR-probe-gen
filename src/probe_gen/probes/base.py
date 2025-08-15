from abc import ABC, abstractmethod
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve


class Probe(ABC):
    @abstractmethod
    def fit(self, train_dataset, validation_dataset, normalize):
        """
        Fits the probe to training data.

        Args:
            train_dataset (dict): train_dataset['X'] has shape [batch_size, dim], train_dataset['y'] has shape [batch_size].
            val_dataset (dict): val_dataset['X'] has shape [batch_size, dim], val_dataset['y'] has shape [batch_size].
            normalize (bool): should the activations be normalized before fitting. 
        
        Returns:
            None
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Get prediction labels (0 or 1) for the dataset.

        Args:
            X (tensor): tensor of aggregated activations with shape [batch_size, dim].
        
        Returns:
            y_pred (tensor): predicted labels of shape [batch_size].
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Get prediction probabilities of each point being class 1 for the dataset.

        Args:
            X (tensor): tensor of aggregated activations with shape [batch_size, dim].
        
        Returns:
            y_pred_proba (tensor): predicted probabilities of shape [batch_size].
        """
        pass

    def eval(self, test_dataset):

        y = self._safe_to_numpy(test_dataset['y'])
        y_pred = self._safe_to_numpy(self.predict(test_dataset['X']))
        y_pred_proba = self._safe_to_numpy(self.predict_proba(test_dataset['X']))

        # Evaluate the model
        accuracy = accuracy_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_pred_proba)

        # Calculate TPR at 1% FPR (as mentioned in paper)
        fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
        target_fpr = 0.01
        idx = np.argmax(fpr >= target_fpr)
        tpr_at_1_fpr = tpr[idx] if idx < len(tpr) else 0

        return {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "tpr_at_1_fpr": tpr_at_1_fpr,
        }, y_pred, y_pred_proba
    
    def _safe_to_numpy(self, data):
        """Convert PyTorch tensor OR numpy array to numpy array"""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data  # Already numpy
        else:
            return np.array(data)  # Convert other types
