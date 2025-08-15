import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from .base import Probe


class SklearnMeanLogisticProbe(Probe):

    def __init__(self, use_bias=True):
        # Create the sklearn classifier model to be optimized.
        self.classifier = LogisticRegression(fit_intercept=use_bias, max_iter=500)
        # Save the normalizing transformation parameters
        self.transformation_mean = 0.0
        self.transformation_std = 1.0
    
    def _mean_aggregation(self, activations, attention_mask):
        return activations.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1).unsqueeze(1)

    def fit(self, train_dataset: dict, validation_dataset: dict, normalize=True) -> None:
        """
        Fits the probe to training data.

        Args:
            train_dataset (dict): train_dataset['X'] has shape [batch_size, seq_len, dim], train_dataset['y'] has shape [batch_size].
            val_dataset (dict): val_dataset['X'] has shape [batch_size, seq_len, dim], val_dataset['y'] has shape [batch_size].
            normalize (bool): should the activations be normalized before fitting. 
        
        Returns:
            None
        """
        if validation_dataset is not None:
            print("Warning: SklearnProbe does not use a validation dataset")
        print("Training probe...")

        X_train = self._mean_aggregation(train_dataset['X'], train_dataset['attention_mask']).detach().cpu().numpy()
        y_train = train_dataset['y'].detach().cpu().numpy()

        # Normalize activations and save the transformation for predicting.
        if normalize:
            self.transformation_mean = np.mean(X_train, axis=0, keepdims=True)  # [1, dim]
            self.transformation_std = np.std(X_train, axis=0, keepdims=True)    # [1, dim]
            X_train = (X_train - self.transformation_mean) / self.transformation_std

        self.classifier.fit(X_train, y_train)
        print("Training complete.")


    def predict(self, X, attention_mask):
        """
        Get prediction labels (0 or 1) for the dataset.

        Args:
            X (tensor): tensor of activations with shape [batch_size, seq_len, dim].
            attention_mask (tensor): tensor indicating which tokens are real [batch_size, seq_len]
        
        Returns:
            y_pred (tensor): predicted labels of shape [batch_size].
        """
        X_numpy = self._mean_aggregation(X, attention_mask).detach().cpu().numpy()
        X_normalized = (X_numpy - self.transformation_mean) / self.transformation_std
        y_pred = self.classifier.predict(X_normalized)
        return y_pred

    def predict_proba(self, X, attention_mask):
        """
        Get prediction probabilities of each point being class 1 for the dataset.

        Args:
            X (tensor): tensor of activations with shape [batch_size, seq_len, dim].
            attention_mask (tensor): tensor indicating which tokens are real [batch_size, seq_len]
        
        Returns:
            y_pred_proba (tensor): predicted probabilities of shape [batch_size].
        """
        X_numpy = self._mean_aggregation(X, attention_mask).detach().cpu().numpy()
        X_normalized = (X_numpy - self.transformation_mean) / self.transformation_std
        y_pred_proba = self.classifier.predict_proba(X_normalized)[:, 1]  # probabilities for class 1
        return y_pred_proba
        

