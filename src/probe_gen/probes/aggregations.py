from abc import ABC, abstractmethod


class Aggregation(ABC):
    @abstractmethod
    def __call__(self, activations):
        """
        Aggregates the activations across tokens into a single representation for the probe to learn from. 

        Args:
            activations (tensor): model activations with shape [batch_size, seq_len, dim].
            attention_mask (tensor): mask indicating which tokens are real (1) vs padding (0) of shape [batch_size, seq_len].
        
        Returns:
            aggregated_activations (tensor): aggregated activations with shape [batch_size, dim].
        """
        pass


# TODO: We need to think about how padding in the input activations tensor should be handled.


class MeanAggregation(Aggregation):
    def __call__(self, activations, attention_mask):
        return activations.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1).unsqueeze(1)