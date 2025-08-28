import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from .base import Probe


class SklearnLogisticProbe(Probe):
    def __init__(self, cfg):
        """
        Initialize the probe.
        Args:
            cfg (ConfigDict): ConfigDict with the following attributes:
                - use_bias (bool): Whether to use bias in the logistic regression model.
                - C (float): The regularization parameter for the logistic regression model.
                - seed (int): The random seed for the logistic regression model.
        """
        super().__init__(cfg)
        # Create the sklearn classifier model to be optimized.
        self.classifier = LogisticRegression(fit_intercept=cfg.use_bias, C=cfg.C, max_iter=500, random_state=cfg.seed)
        # Save the normalizing transformation parameters
        self.transformation_mean = 0.0
        self.transformation_std = 1.0

    def fit(self, train_dataset: dict, validation_dataset: dict) -> None:
        """
        Fits the probe to training data.
        Args:
            train_dataset (dict): train_dataset['X'] has shape [batch_size, dim], train_dataset['y'] has shape [batch_size].
            val_dataset (dict): val_dataset['X'] has shape [batch_size, dim], val_dataset['y'] has shape [batch_size].
        Returns:
            None
        """
        if validation_dataset is not None:
            print("Warning: SklearnProbe does not use a validation dataset")
        print("Training probe...")

        X_train = train_dataset['X'].detach().cpu().numpy()
        y_train = train_dataset['y'].detach().cpu().numpy()

        # Normalize activations and save the transformation for predicting.
        if self.cfg.normalize:
            self.transformation_mean = np.mean(X_train, axis=0, keepdims=True)  # [1, dim]
            self.transformation_std = np.std(X_train, axis=0, keepdims=True)    # [1, dim]
            X_train = (X_train - self.transformation_mean) / self.transformation_std

        self.classifier.fit(X_train, y_train)

    def predict(self, X):
        """
        Get prediction labels (0 or 1) for the dataset.
        Args:
            X (tensor): tensor of aggregated activations with shape [batch_size, dim].
        Returns:
            y_pred (tensor): predicted labels of shape [batch_size].
        """
        X_numpy = X.detach().cpu().numpy()
        X_normalized = (X_numpy - self.transformation_mean) / self.transformation_std
        y_pred = self.classifier.predict(X_normalized)
        return y_pred

    def predict_proba(self, X):
        """
        Get prediction probabilities of each point being class 1 for the dataset.
        Args:
            X (tensor): tensor of aggregated activations with shape [batch_size, dim].
        Returns:
            y_pred_proba (tensor): predicted probabilities of shape [batch_size].
        """
        X_numpy = X.detach().cpu().numpy()
        X_normalized = (X_numpy - self.transformation_mean) / self.transformation_std
        y_pred_proba = self.classifier.predict_proba(X_normalized)[:, 1]  # probabilities for class 1
        return y_pred_proba
        

