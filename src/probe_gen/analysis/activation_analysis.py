import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


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










def plot_activations_pca(dataset1, dataset2, labels_1, labels_2, n_components=2, dataset1_name="Dataset 1", dataset2_name="Dataset 2", figsize=(10, 8), alpha=0.7):
    """
    Perform PCA on combined datasets and visualize with different colors for each dataset.
    
    Args:
        dataset1: torch.Tensor of shape [n1, d] - first dataset
        dataset2: torch.Tensor of shape [n2, d] - second dataset  
        n_components: int - number of PCA components (2 for 2D plot, 3 for 3D plot)
        dataset1_name: str - label for first dataset
        dataset2_name: str - label for second dataset
        figsize: tuple - figure size for matplotlib
        alpha: float - alpha value with which to plot each point
    
    Returns:
        pca: fitted PCA object
        transformed_data: numpy array of shape [n1+n2, n_components] - PCA transformed data
        labels: numpy array indicating which dataset each point belongs to
    """
    
    # Convert to numpy if needed and combine datasets
    if isinstance(dataset1, torch.Tensor):
        dataset1 = dataset1.detach().cpu().numpy()
    if isinstance(dataset2, torch.Tensor):
        dataset2 = dataset2.detach().cpu().numpy()
    
    # Combine the datasets
    combined_data = np.vstack([dataset1, dataset2])
    n1, n2 = len(dataset1), len(dataset2)
    
    # Create labels (0 for dataset1, 1 for dataset2)
    labels = np.concatenate([np.zeros(n1), np.ones(n2)])
    
    # Perform PCA on combined data
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(combined_data)
    
    # Split back into separate datasets for plotting
    dataset1_pca = transformed_data[:n1]
    dataset2_pca = transformed_data[n1:]

    dataset1_pca_pos = dataset1_pca[torch.tensor(labels_1, dtype=torch.bool)]
    dataset1_pca_neg = dataset1_pca[~torch.tensor(labels_1, dtype=torch.bool)]
    dataset2_pca_pos = dataset2_pca[torch.tensor(labels_2, dtype=torch.bool)]
    dataset2_pca_neg = dataset2_pca[~torch.tensor(labels_2, dtype=torch.bool)]

    
    # Create the plot
    if n_components == 2:
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.scatter(dataset1_pca_pos[:, 0], dataset1_pca_pos[:, 1], 
                             alpha=alpha, label=f"{dataset1_name} - Positive", s=50, color="red")
        ax.scatter(dataset1_pca_neg[:, 0], dataset1_pca_neg[:, 1], 
                             alpha=alpha, label=f"{dataset1_name} - Negative", s=50, color="orange")
        
        ax.scatter(dataset2_pca_pos[:, 0], dataset2_pca_pos[:, 1], 
                             alpha=alpha, label=f"{dataset2_name} - Positive", s=50, color="blue")
        ax.scatter(dataset2_pca_neg[:, 0], dataset2_pca_neg[:, 1], 
                             alpha=alpha, label=f"{dataset2_name} - Negative", s=50, color="cyan")
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title('PCA Visualization of Two Datasets')
        ax.legend()
        ax.grid(True, alpha=alpha)
        
    elif n_components == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot both datasets with different colors
        ax.scatter(dataset1_pca[:, 0], dataset1_pca[:, 1], dataset1_pca[:, 2],
                  alpha=alpha, label=dataset1_name, s=50)
        ax.scatter(dataset2_pca[:, 0], dataset2_pca[:, 1], dataset2_pca[:, 2],
                  alpha=alpha, label=dataset2_name, s=50)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
        ax.set_title('3D PCA Visualization of Two Datasets')
        ax.legend()
        
    else:
        raise ValueError("n_components must be 2 or 3 for visualization")
    
    plt.tight_layout()
    plt.show()
    
    # Print some useful information
    print(f"Dataset 1 shape: {dataset1.shape}")
    print(f"Dataset 2 shape: {dataset2.shape}")
    print(f"Combined dataset shape: {combined_data.shape}")
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.1%}")
    
    return pca, transformed_data, labels