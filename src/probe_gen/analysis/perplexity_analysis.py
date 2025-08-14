import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
import numpy as np


def save_perplexities(perplexities, model_name, dataset_name):
    wandb.init(project="LASR_probe_gen", entity="samdower")
    np.save("perplexities.npy", perplexities)
    artifact = wandb.Artifact(f"perplexities-{model_name}-{dataset_name}", type="results")
    artifact.add_file("perplexities.npy")
    wandb.log_artifact(artifact)
    wandb.finish()

def load_perplexities(model_name, dataset_name):
    api = wandb.Api()
    artifact = api.artifact(f'samdower/LASR_probe_gen/perplexities-{model_name}-{dataset_name}:latest')
    artifact_dir = artifact.download()
    arr = np.load(f"{artifact_dir}/perplexities.npy")
    return arr









def plot_perplexities(perplexities_list, labels, use_log_scale=True, remove_outliers=False):

    def remove_large_outliers(data):
        arr = np.array(data)
        # Calculate the 99th percentile (threshold for top 1%)
        threshold = np.percentile(arr, 99)
        # Keep only values below this threshold
        filtered_arr = arr[arr < threshold]

        return filtered_arr

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 1, 1)

    for i in range(len(perplexities_list)):
        perplexities = perplexities_list[i]
        if remove_outliers:
            perplexities = remove_large_outliers(perplexities)

        if use_log_scale:
            hist, bins = np.histogram(perplexities, bins=20)
            logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
            plt.hist(perplexities, alpha=0.5, label=labels[i], bins=logbins)
            plt.xscale('log')
        else:
            plt.hist(perplexities, alpha=0.5, label=labels[i], bins=20)

    # plt.axvline(x = mean_on_distance, color = 'blue', linestyle='dashed', label = 'on_policy_mean')
    # plt.axvline(x = mean_off_distance, color = 'orange', linestyle='dashed', label = 'off_policy_mean')
    plt.xlabel('Perplexity')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Perplexities')

    plt.tight_layout()
    plt.show()










def _calculate_single_response_perplexity(model, tokenizer, input_text, response_text):
    """
    Calculate perplexity of a model response given the input.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        input_text: str - The original input/prompt
        response_text: str - The model's response to evaluate
    
    Returns:
        float: Perplexity of the response
    """
    
    # Tokenize input and response separately
    input_ids = tokenizer(input_text, return_tensors="pt", add_special_tokens=True)["input_ids"]
    response_ids = tokenizer(response_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    
    # Combine input and response
    full_sequence = torch.cat([input_ids, response_ids], dim=1).to(model.device).to(torch.int64)
    
    # Calculate the number of tokens in input vs response
    input_length = input_ids.shape[1]
    response_length = response_ids.shape[1]
    
    # Get model outputs for the full sequence
    with torch.no_grad():
        outputs = model(full_sequence)
        logits = outputs.logits  # Shape: [batch_size, sequence_length, vocab_size]
    
    # We want to evaluate the response tokens, so we need:
    # - logits for positions [input_length-1 : input_length+response_length-1]
    # - targets for positions [input_length : input_length+response_length]
    
    # Get logits for predicting response tokens
    response_logits = logits[0, input_length-1:input_length+response_length-1, :]  # [response_length, vocab_size]
    
    # Get target tokens (the actual response tokens)
    response_targets = full_sequence[0, input_length:input_length+response_length]  # [response_length]
    
    # Calculate cross-entropy loss for response tokens only
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fn(response_logits, response_targets)
    
    # Convert to perplexity
    perplexity = torch.exp(loss).item()
    
    return perplexity

def calculate_response_perplexities_sequentially(model, tokenizer, input_text_list, response_text_list, verbose=False):
    perplexities = []
    for i in range(len(input_text_list)):
        if verbose and i % 500 == 0: 
            print(f"processing response {i}")
        perplexity = _calculate_single_response_perplexity(model, tokenizer, input_text_list[i], response_text_list[i])
        perplexities.append(perplexity)
    return perplexities



def calculate_response_perplexities_batched(model, tokenizer, input_texts, response_texts):
    """
    Calculate perplexities for a batch of responses using padding.
    This is the most straightforward but potentially less efficient approach.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        input_texts: List of input texts
        response_texts: List of corresponding response texts
    
    Returns:
        List[float]: Perplexities for each response
    """
    
    batch_size = len(input_texts)
    assert len(response_texts) == batch_size
    
    # Tokenize all sequences
    full_sequences = []
    input_lengths = []
    response_lengths = []
    
    for input_text, response_text in zip(input_texts, response_texts):
        input_ids = tokenizer(input_text, add_special_tokens=True)["input_ids"]
        response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
        full_seq = input_ids + response_ids
        
        full_sequences.append(full_seq)
        input_lengths.append(len(input_ids))
        response_lengths.append(len(response_ids))
    
    # Pad sequences to the same length
    max_length = max(len(seq) for seq in full_sequences)
    
    padded_sequences = []
    attention_masks = []
    
    for seq in full_sequences:
        padded = seq + [tokenizer.pad_token_id] * (max_length - len(seq))
        mask = [1] * len(seq) + [0] * (max_length - len(seq))
        
        padded_sequences.append(padded)
        attention_masks.append(mask)
    
    # Convert to tensors
    input_ids = torch.tensor(padded_sequences).to(model.device)
    attention_mask = torch.tensor(attention_masks).to(model.device)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
    
    # Calculate perplexities for each sequence
    perplexities = []
    
    for i in range(batch_size):
        input_len = input_lengths[i]
        response_len = response_lengths[i]
        
        if response_len == 0:
            perplexities.append(float('inf'))
            continue
        
        # Get logits and targets for this response
        response_logits = logits[i, input_len-1:input_len+response_len-1, :]
        response_targets = input_ids[i, input_len:input_len+response_len]
        
        # Calculate loss
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fn(response_logits, response_targets)
        perplexity = torch.exp(loss).item()
        perplexities.append(perplexity)
    
    return perplexities