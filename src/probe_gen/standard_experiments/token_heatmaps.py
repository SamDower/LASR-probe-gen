import json

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Load the JSONL file that contains questions, model outputs, and labels from HuggingFace
from huggingface_hub import hf_hub_download
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import probe_gen.probes as probes
from probe_gen.config import ConfigDict
from probe_gen.gen_data.utils import format_prompts_from_pairs


def train_probe_on(probe_type, behaviour, datasource, activations_model, generation_method, response_model, mode, verbose=False):

    # Load the best hyperparameters or set your own
    cfg = ConfigDict.from_json(activations_model, probe_type, behaviour)
    if verbose:
        print("Loaded hyperparameters:")
        print(f"  Layer: {cfg.layer}")
        print(f"  Use bias: {cfg.use_bias}")
        print(f"  Normalize: {cfg.normalize}")
        if hasattr(cfg, 'C'):
            print(f"  C (inverse ofregularization): {cfg.C}")

    # Load activations and labels from HuggingFace
    activations_tensor, attention_mask, labels_tensor = probes.load_hf_activations_at_layer(
        behaviour, 
        datasource, 
        activations_model, 
        response_model, 
        generation_method, 
        mode, 
        cfg.layer, 
        and_labels=True, 
        verbose=verbose
    )

    if verbose:
        print("\nLoaded data:")
        print(f"  Activations shape: {activations_tensor.shape}")
        print(f"  Attention mask shape: {attention_mask.shape}")
        print(f"  Labels shape: {labels_tensor.shape}")
        print(f"  Positive samples: {labels_tensor.sum().item():.0f}")
        print(f"  Negative samples: {(len(labels_tensor) - labels_tensor.sum()).item():.0f}")


    # Aggregate activations if using mean probe
    if "mean" in probe_type:
        activations_aggregated = probes.MeanAggregation()(activations_tensor, attention_mask)
        if verbose:
            print(f"Aggregated activations shape: {activations_aggregated.shape}")
    else:
        activations_aggregated = activations_tensor
        if verbose:
            print(f"Using full sequence activations: {activations_aggregated.shape}")

    # Create train and validation datasets
    split_val = 2500 if behaviour in ["deception", "sandbagging"] else 3500

    train_dataset, val_dataset, test_dataset = probes.create_activation_datasets(
        activations_aggregated, labels_tensor, splits=[split_val, 500, 0], verbose=True)

    # Initialize probe
    if probe_type == "mean":
        probe = probes.SklearnLogisticProbe(cfg)
    elif probe_type == "mean_torch":
        probe = probes.TorchLinearProbe(cfg)
    elif probe_type == "attention_torch":
        probe = probes.TorchAttentionProbe(cfg)

    # Train the probe
    if verbose:
        print("\nTraining probe...")
    probe.fit(train_dataset, val_dataset)

    # Evaluate on validation set
    if verbose:
        eval_dict, y_pred, y_pred_proba = probe.eval(val_dataset)
        print(f'\n✓ Validation ROC-AUC: {eval_dict["roc_auc"]:.4f}')
        print(f'  Accuracy: {eval_dict["accuracy"]:.4f}')
    
    return probe



def extract_probe_weights(probe):
    # Extract probe weights
    if isinstance(probe, probes.SklearnLogisticProbe):
        # For sklearn probe
        probe_weights = probe.classifier.coef_[0]  # Shape: (hidden_dim,)
        probe_bias = probe.classifier.intercept_[0]        
    elif isinstance(probe, probes.TorchLinearProbe):
        # For PyTorch mean probes
        probe_weights = probe.model.linear.weight.detach().cpu().numpy()[0]  # Shape: (hidden_dim,)
        probe_bias = probe.model.linear.bias.detach().cpu().numpy()[0]
    elif isinstance(probe, probes.TorchAttentionProbe):
        # For attention probe, get the query/key/value weights
        probe_weights = {
            'W_Q': probe.model.W_Q.detach().cpu().numpy(),
            'W_K': probe.model.W_K.detach().cpu().numpy(),
            'W_V': probe.model.W_V.detach().cpu().numpy(),
            'W_out': probe.model.W_out.detach().cpu().numpy()
        }
        probe_bias = None
    else:
        raise ValueError(f"Unsupported probe type: {type(probe)}")

    return probe_weights, probe_bias



def load_labelled_responses_and_activations(probe_type, behaviour, datasource, activations_model, generation_method, response_model, mode, verbose=False):

    generation_method_for_labels = generation_method.replace("_included", "")
    if generation_method == "off_policy":
        generation_method = "on_policy"
    labels_filename = f"{datasource}/{response_model}_{generation_method_for_labels}_{mode}.jsonl"

    # Download from HuggingFace
    repo_id = f"lasrprobegen/{behaviour}-activations"

    labels_localpath = hf_hub_download(
        repo_id=repo_id,
        filename=labels_filename,
        repo_type="dataset"
    )

    # Load the best hyperparameters or set your own
    cfg = ConfigDict.from_json(activations_model, probe_type, behaviour)

    # Load activations and labels from HuggingFace
    activations_tensor, attention_mask, labels_tensor = probes.load_hf_activations_at_layer(
        behaviour, 
        datasource, 
        activations_model, 
        response_model, 
        generation_method, 
        mode, 
        cfg.layer, 
        and_labels=True, 
        verbose=verbose
    )

    # Load the JSONL file
    data_rows = []
    with open(labels_localpath, 'r') as file:
        for line in file:
            data_dict = json.loads(line)
            data_rows.append(data_dict)

    # Convert to DataFrame for easier manipulation
    dataset_df = pd.DataFrame(data_rows)

    # if 'input' in dataset_df.columns:
    #     print(f"  Input: {dataset_df.iloc[0]['input'][:200]}...")
    # if 'model_outputs' in dataset_df.columns:
    #     print(f"  Output: {dataset_df.iloc[0]['model_outputs'][:200]}...")
    # if 'scale_labels' in dataset_df.columns:
    #     print(f"  Label: {dataset_df.iloc[0]['scale_labels']}")
    
    return dataset_df, activations_tensor, attention_mask, labels_tensor





def get_probe_scores_and_tokens(tokenizer, index, responses_df, activations_tensor, probe):

    inputs = json.loads(responses_df.iloc[index]["inputs"])
    human = inputs[0]["content"]
    assistant = inputs[1]["content"]
    formatted_pairs = format_prompts_from_pairs(tokenizer, [human], [assistant]) # List of strings

    inputs = tokenizer(
        formatted_pairs,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=None,
    )

    # Convert the tensor of IDs back to readable token strings
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    probe_weights, probe_bias = extract_probe_weights(probe)

    # Compute scores
    scores = []
    for i in range(len(tokens)):
        activation = activations_tensor[index][i]
        score = np.dot(activation.numpy(), probe_weights) + probe_bias
        scores.append(score)
        tokens[i] = tokens[i].replace('Ġ', ' ').replace('Ċ', '\\n')
    
    return scores, tokens





def visualize_token_heatmap(
    tokenizer,
    index,
    responses_df,
    activations_tensor,
    probe,
    labels_tensor,
    behaviour,
    figsize=(20, 12),
    ):

    """Create a matplotlib heatmap of token probabilities for the ENTIRE question and answer"""

    label = labels_tensor[index].item()
    predicted_label = probe.predict(activations_tensor[index].mean(dim=0).unsqueeze(0))[0].item()
    scores, tokens = get_probe_scores_and_tokens(tokenizer, index, responses_df, activations_tensor, probe)
    
    # Create custom colormap: red (0) -> white (0.5) -> blue (1)
    colors = ['#ff4444', '#ffffff', '#4444ff']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('prob_cmap', colors, N=n_bins)
    
    # Create normalizer centered at 0 with actual data range
    norm = TwoSlopeNorm(vmin=min(scores), vcenter=0, vmax=max(scores))
    
    # Dynamically calculate figure height based on number of tokens and wrapping
    max_width = 100  # Maximum width before wrapping
    estimated_rows = sum(len(t) * 0.6 + 0.1 for t in tokens) / max_width + 1
    dynamic_height = max(12, estimated_rows * 1.2)  # At least 12, scale with content
    
    # Create figure with dynamic height
    fig, ax = plt.subplots(figsize=(figsize[0], dynamic_height))
    
    # Plot tokens as colored rectangles
    x_pos = 0
    y_pos = 0
    rect_height = 0.4
    
    for i, (token, prob) in enumerate(zip(tokens, scores)):

        # Determine width based on token length
        token_width = max(len(token) * 0.6, 0.5)

        # Wrap to new line if needed
        if x_pos + token_width > max_width:
            x_pos = 0
            y_pos -= rect_height + 0.1
        
        # Color based on probability
        color = cmap(norm(prob))
        
        # Draw rectangle
        rect = mpatches.Rectangle((x_pos, y_pos), token_width, rect_height, 
                                  facecolor=color, edgecolor='gray', linewidth=0.5)
        ax.add_patch(rect)
        
        # Add text
        ax.text(x_pos + token_width/2, y_pos + rect_height/2, token, 
               ha='center', va='center', fontsize=11, 
               fontfamily='monospace', weight='bold') #, weight='bold' if section == 'Q' else 'normal')
        
        x_pos += token_width + 0.1
    
    # Set axis properties with enough vertical space
    ax.set_xlim(0, max_width)
    ax.set_ylim(y_pos - 1.5, 1)  # Extra space at bottom
    ax.axis('off')
    
    # Add title and legend
    title = f'Token-Level Probe Probabilities - {behaviour} (Label={label}, Prediction={predicted_label})\n'
    title += f'{len(tokens)} tokens'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Create colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.02, aspect=50)
    cbar.set_label('Probe score', fontsize=12, fontweight='bold')

    # Create ticks that span both negative and positive ranges
    negative_ticks = np.linspace(min(scores), 0, num=5)  # 5 ticks from min to 0
    positive_ticks = np.linspace(0, max(scores), num=5)[1:]  # 4 ticks from 0 to max (excluding 0)
    tick_values = np.concatenate([negative_ticks, positive_ticks])
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f'{val:.1f}' for val in tick_values])

    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.show()



def get_indices_for_each_prediction_type(activations_tensor, attention_mask, labels_tensor, probe):
    """Get indices of correctly classified examples for a given label"""

    predicted_labels = torch.tensor(probe.predict(probes.MeanAggregation()(activations_tensor, attention_mask)))

    # True Positives: predicted 1, actual 1
    tp_indices = ((predicted_labels == 1) & (labels_tensor == 1)).nonzero(as_tuple=True)[0]

    # False Positives: predicted 1, actual 0
    fp_indices = ((predicted_labels == 1) & (labels_tensor == 0)).nonzero(as_tuple=True)[0]

    # True Negatives: predicted 0, actual 0
    tn_indices = ((predicted_labels == 0) & (labels_tensor == 0)).nonzero(as_tuple=True)[0]

    # False Negatives: predicted 0, actual 1
    fn_indices = ((predicted_labels == 0) & (labels_tensor == 1)).nonzero(as_tuple=True)[0]
    
    return tp_indices, fp_indices, fn_indices, tn_indices