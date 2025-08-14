import torch
import matplotlib.pyplot as plt


def project_onto_line_batch(X, A, B):
    """
    Project a batch of vectors X onto the line connecting A and B.
    
    Args:
        X: torch.Tensor of shape [n, d] - batch of vectors to project
        A: torch.Tensor of shape [d] - start point of line
        B: torch.Tensor of shape [d] - end point of line
    
    Returns:
        distances: torch.Tensor of shape [n] - absolute distances along line from A
                  distances[i]=0 means X[i]'s projection is at A
                  distances[i]>0 means X[i]'s projection is in direction of B
                  distances[i]<0 means X[i]'s projection is in opposite direction from B
                  distances[i]=||B-A|| means X[i]'s projection is at B
    """
    # Vector from A to B - shape [d]
    AB = B - A
    
    # Vector from A to each X - shape [n, d]
    AX = X - A.unsqueeze(0)  # A.unsqueeze(0) broadcasts A to shape [1, d]
    
    # Project each AX onto AB using batched dot product
    # AX @ AB gives us the dot product for each vector in the batch
    # AB @ AB gives us ||AB||^2
    numerator = torch.sum(AX * AB.unsqueeze(0), dim=1)  # shape [n]
    denominator = torch.dot(AB, AB)  # scalar
    
    # Get relative parameters for each vector
    t_batch = numerator / denominator  # shape [n]
    
    # Convert to absolute distances by multiplying by length of AB
    AB_length = torch.norm(AB)
    distances = t_batch * AB_length
    
    return distances


def plot_activations_mean_line_projections(activations_1, activations_2, labels, layer):
    mean_1 = activations_1.mean(dim=0)
    mean_2 = activations_2.mean(dim=0)

    distances_1 = project_onto_line_batch(activations_1, mean_1, mean_2)
    distances_2 = project_onto_line_batch(activations_2, mean_1, mean_2)

    mean_distance_1 = project_onto_line_batch(mean_1.unsqueeze(0), mean_1, mean_2)[0]
    mean_distance_2 = project_onto_line_batch(mean_2.unsqueeze(0), mean_1, mean_2)[0]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 1, 1)
    plt.hist(distances_1, alpha=0.5, label=labels[0], bins=40)
    plt.hist(distances_2, alpha=0.5, label=labels[1], bins=40)
    plt.axvline(x = mean_distance_1, color = 'blue', linestyle='dashed', label = f'{labels[0]} mean')
    plt.axvline(x = mean_distance_2, color = 'orange', linestyle='dashed', label = f'{labels[1]} mean')
    plt.xlabel(f'Distance from mean of {labels[0]}')
    plt.ylabel('Count')
    plt.legend()
    plt.title(f'Activation projections onto mean line (layer {layer})')

    plt.tight_layout()
    plt.show()
