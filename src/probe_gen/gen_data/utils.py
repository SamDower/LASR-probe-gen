import gc
import json
import os
import pickle
import joblib
import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def suggest_layer(total_layers):
    # Sweet spot seems to be 60-70% through
    return int(total_layers * 0.65)


def parse_layers_arg(layers_str, total_layers):
    """Parse the layers argument string into a list of layer indices.

    Args:
        layers_str: String specifying layers. Can be:
            - "all": Use all layers
            - "auto": Use automatic selection (65% through)
            - comma-separated indices: "0,5,10" -> [0, 5, 10]
            - range notation: "5-10" -> [5, 6, 7, 8, 9, 10]
            - mixed: "0,5-8,15" -> [0, 5, 6, 7, 8, 15]
        total_layers: Total number of layers in the model

    Returns:
        List of layer indices to extract
    """
    if layers_str == "all":
        return list(range(total_layers))
    elif layers_str == "auto":
        start_idx = suggest_layer(total_layers)
        return list(range(start_idx, total_layers))

    # Parse comma-separated values that can include ranges
    layer_indices = []
    parts = layers_str.split(",")

    for part in parts:
        part = part.strip()
        if "-" in part:
            # Handle range notation like "5-10"
            start, end = part.split("-")
            start, end = int(start.strip()), int(end.strip())
            layer_indices.extend(range(start, end + 1))
        else:
            # Handle single index
            layer_indices.append(int(part))

    # Remove duplicates and sort
    layer_indices = sorted(list(set(layer_indices)))

    # Validate indices
    for idx in layer_indices:
        if idx < 0 or idx >= total_layers:
            raise ValueError(
                f"Layer index {idx} is out of range. Model has {total_layers} layers (0-{total_layers - 1})"
            )

    return layer_indices


def _cleanup_gpu_memory():
    """Clean up GPU memory if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# * model utils
def get_pad_token(model_id, tokenizer):
    if tokenizer.pad_token is not None:
        return tokenizer.pad_token
    model_to_pad_token = {
        "meta-llama/Llama-3.2-3B-Instruct": "<|finetune_right_pad_id|>",
        "google/gemma-7b-it": "<pad>",
        "mistralai/Ministral-8B-Instruct-2410": "<pad>",
        "meta-llama/Llama-3.1-8B-Instruct": "<|finetune_right_pad_id|>",
    }
    pad_token = model_to_pad_token[model_id]
    # print(f"Using pad token for {model_id}: {pad_token}")
    return pad_token


def get_model(model_name: str):
    # Load the model with float16 precision
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Added float16 precision
        device_map="auto",
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set padding side to left for decoder-only models to avoid warnings
    tokenizer.padding_side = "left"

    # Set the pad token using the same logic as control experiments
    tokenizer.pad_token = get_pad_token(model_name, tokenizer)
    # print(f"Pad token set to: {tokenizer.pad_token}")
    # print(f"Padding side set to: {tokenizer.padding_side}")

    return model, tokenizer


def get_res_layers_to_enumerate(model):
    model_name = model.config._name_or_path
    if "gpt" in model_name:
        return model.transformer.h
    elif "pythia" in model_name:
        return model.gpt_neox.layers
    elif "bert" in model_name:
        return model.encoder.layer
    elif "mistral" in model_name:
        return model.model.layers
    elif "gemma" in model_name:
        return model.model.layers
    elif "llama" in model_name:
        return model.model.layers
    elif "qwen" in model_name.lower():
        return model.model.layers
    else:
        raise ValueError(f"Unsupported model: {model_name}.")


# * chat templates
def format_prompts_from_strings(tokenizer, prompt_strings):
    """
    Takes a list of string prompts and formats them for the model
    Input: ["how do you do", "what is your favorite condiment", ...]
    Output: List of properly formatted prompts
    """
    formatted_prompts = []
    for prompt_string in prompt_strings:
        # Convert string to message format
        messages = [{"role": "user", "content": prompt_string}]
        # Apply chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(formatted_prompt)
    return formatted_prompts


def format_prompts_from_pairs(tokenizer, human_list, assistant_list):
    """
    Takes lists of human and assistant strings and formats them as a single dialogue turn.
    Returns chat-formatted strings without a generation prompt (teacher forcing / off-policy).
    """
    assert len(human_list) == len(assistant_list), (
        "human_list and assistant_list must be same length"
    )
    formatted_prompts = []
    for human, assistant in zip(human_list, assistant_list):
        messages = [
            {"role": "user", "content": human},
            {"role": "assistant", "content": assistant},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        formatted_prompts.append(formatted_prompt)
    return formatted_prompts


def _prepare_batch_inputs(tokenizer, prompts, max_length=512):
    """Prepare and tokenize batch inputs."""
    # Handle single prompt case
    if isinstance(prompts, str):
        prompts = [prompts]

    return tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_length,
    )


# * hook utils
def _generate_sequences(model, tokenizer, inputs, max_new_tokens=75, temperature=1.0):
    """
    Generate sequences from inputs with smart temperature handling.

    Args:
        model: The language model
        tokenizer: Model tokenizer
        inputs: Tokenized inputs
        max_new_tokens: Maximum new tokens to generate
        temperature: Generation temperature (0.0 for greedy, >0.0 for sampling)

    Returns:
        Generated token sequences
    """
    model.eval()
    with torch.no_grad():
        # Prepare base generation arguments
        pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        base_args = {**inputs, "max_new_tokens": max_new_tokens, "pad_token_id": pad_id}

        # Smart temperature handling with clean parameter passing
        if temperature == 0.0:
            # Use greedy decoding - only pass necessary parameters (no temperature/top_p)
            generation_args = {
                **base_args,
                "do_sample": False,  # Greedy decoding
            }
        else:
            # Use sampling with temperature - only pass sampling parameters when needed
            generation_args = {
                **base_args,
                "do_sample": True,
                "temperature": temperature,
            }

        outputs = model.generate(**generation_args)  # this gives input + output
    return outputs


def _create_activation_hook(activations_dict, layer_name, verbose=False):
    """Create a hook function to capture layer activations."""

    def hook(model, input, output):
        # Extract activation tensor from output
        activation_tensor = output[0] if isinstance(output, tuple) else output

        if verbose:
            print(
                f"Layer {layer_name}: output type={type(activation_tensor)}, shape={activation_tensor.shape}"
            )
            print(
                f"  Batch size: {activation_tensor.shape[0]}, "
                f"Sequence length: {activation_tensor.shape[1]}, "
                f"Hidden size: {activation_tensor.shape[2]}"
            )

        activations_dict[layer_name] = activation_tensor.detach().cpu()

    return hook


def _register_activation_hooks(model, layers_str="auto", verbose=False):
    """Register hooks on specified model layers and return the hooks and activations dict.

    Args:
        model: The model to register hooks on
        layers_str: String specifying which layers to extract (default: "auto" for 65% through)
        verbose: Whether to print debug information
    """
    activations = {}
    layers_to_enum = get_res_layers_to_enumerate(model)
    hooks = []

    total_layers = len(layers_to_enum)
    layer_indices = parse_layers_arg(layers_str, total_layers)

    if verbose:
        print(f"Total layers: {total_layers}")
        print(f"Extracting from layers: {layer_indices}")

    for layer_idx in layer_indices:
        layer = layers_to_enum[layer_idx]
        hook_fn = _create_activation_hook(activations, layer_idx, verbose)
        hook_handle = layer.register_forward_hook(hook_fn)
        hooks.append(hook_handle)

    return hooks, activations


def _decode_outputs(tokenizer, outputs, verbose=False):
    """Decode generated token sequences to text."""
    raw_outputs = []
    for i, output in enumerate(outputs):
        if verbose:
            print(f"Raw token sequence length: {len(output)}")

        full_decoded = tokenizer.decode(output, skip_special_tokens=True)
        if verbose:
            print(f"Decoded length: {len(full_decoded)}")

        raw_outputs.append(full_decoded)

    return raw_outputs


def get_batch_outputs_only(
    model,
    tokenizer,
    prompts,
    outputs=None,
    verbose=False,
    max_length=512,
    max_new_tokens=75,
    temperature=1.0,
    do_generation=True,
):
    """
    Efficient batch version - generate outputs from prompts without extracting activations.

    Args:
        model: The language model
        tokenizer: Model tokenizer
        prompts: String or list of strings
        verbose: Whether to print debug information
        max_length: Maximum input sequence length
        max_new_tokens: Maximum new tokens to generate
        temperature: Generation temperature
            - 0.0: Uses greedy decoding (deterministic, most likely tokens)
            - 0.1-1.0: Low to moderate randomness (good for most tasks)
            - 1.0-2.0: High randomness (creative generation)
            - >2.0: Very high randomness (automatically clamped to 2.0)
        outputs: If provided, use these outputs instead of generating new ones (on policy vs. off policy)

    Returns:
        tuple: (decoded_outputs, input_length)
            - decoded_outputs: List of generated text strings
            - input_length: Original input sequence length
    """
    # Step 1: Prepare inputs
    inputs = _prepare_batch_inputs(tokenizer, prompts, max_length)
    inputs = inputs.to(model.device)

    input_length = inputs.input_ids.shape[1]
    batch_size = inputs.input_ids.shape[0]

    # Step 2: Build sequences (input_ids covering input [+ generated tokens])
    if outputs is not None:
        sequences = outputs
    elif do_generation:
        sequences = _generate_sequences(
            model, tokenizer, inputs, max_new_tokens, temperature
        )  # tensor shape: [batch, input_len + new_len]
    else:
        sequences = inputs.input_ids

    # Clean up input tensors
    del inputs
    _cleanup_gpu_memory()

    # Step 3: Decode outputs (no activation extraction needed)
    # Extract only the newly generated tokens (not the input prompt)
    if do_generation and isinstance(sequences, torch.Tensor):
        # For generation, slice off the input tokens to get only new tokens
        new_token_sequences = sequences[:, input_length:]
        raw_outputs = _decode_outputs(tokenizer, new_token_sequences, verbose)
    else:
        # For non-generation (off-policy), decode the full sequence
        raw_outputs = _decode_outputs(tokenizer, sequences, verbose)

    if verbose:
        print("\nBatch processing summary:")
        print(f"  Processed {batch_size} prompts")
        print(f"  Original input length: {input_length}")
        if isinstance(sequences, torch.Tensor):
            print(f"  Final sequence length: {sequences.shape[1]}")

    # Final cleanup
    del sequences
    _cleanup_gpu_memory()

    return raw_outputs, input_length


def get_batch_res_activations_with_generation(
    model,
    tokenizer,
    prompts,
    outputs=None,
    verbose=False,
    max_length=512,
    max_new_tokens=75,
    temperature=1.0,
    do_generation=True,
    layers_str="auto",
):
    """
    Efficient batch version - get activations from all layers for multiple prompts.
    Now captures activations for the FULL sequence (input + generated tokens).

    Args:
        model: The language model
        tokenizer: Model tokenizer
        prompts: String or list of strings
        verbose: Whether to print debug information
        max_length: Maximum input sequence length
        max_new_tokens: Maximum new tokens to generate
        temperature: Generation temperature
            - 0.0: Uses greedy decoding (deterministic, most likely tokens)
            - 0.1-1.0: Low to moderate randomness (good for most tasks)
            - 1.0-2.0: High randomness (creative generation)
            - >2.0: Very high randomness (automatically clamped to 2.0)
        outputs: If provided, use these outputs instead of generating new ones (on policy vs. off policy)
        do_generation: Whether to generate new tokens (True) or use existing sequences (False)
        layers_str: String specifying which layers to extract (default: "auto" for 65% through)

    Returns:
        tuple: (activations_dict, decoded_outputs, input_length)
            - activations_dict: Dict mapping layer indices to activation tensors
            - decoded_outputs: List of generated text strings
            - input_length: Original input sequence length
    """
    # Step 1: Prepare inputs
    inputs = _prepare_batch_inputs(tokenizer, prompts, max_length)
    inputs = inputs.to(model.device)

    input_length = inputs.input_ids.shape[1]
    batch_size = inputs.input_ids.shape[0]

    # Step 2: Build sequences (input_ids covering input [+ generated tokens])
    if outputs is not None:
        sequences = outputs
    elif do_generation:
        sequences = _generate_sequences(
            model, tokenizer, inputs, max_new_tokens, temperature
        )  # tensor shape: [batch, input_len + new_len]
    else:
        sequences = inputs.input_ids

    # Clean up input tensors
    del inputs
    _cleanup_gpu_memory()

    # Step 3: Register hooks and capture activations
    hooks, activations = _register_activation_hooks(model, layers_str, verbose)

    try:
        # Forward pass to capture activations
        with torch.no_grad():
            # Build attention mask from pad token id to safely handle padding
            if isinstance(sequences, torch.Tensor):
                attention_mask = (
                    sequences.ne(tokenizer.pad_token_id).long().to(sequences.device)
                )
                with torch.no_grad():
                    model(input_ids=sequences, attention_mask=attention_mask)
            else:
                # Fallback for unexpected type (should not happen)
                with torch.no_grad():
                    model(sequences)
    finally:
        # Always clean up hooks
        for hook in hooks:
            hook.remove()

    # Step 4: Decode outputs
    raw_outputs = _decode_outputs(tokenizer, sequences, verbose)

    if verbose:
        print("\nBatch processing summary:")
        print(f"  Processed {batch_size} prompts")
        print(f"  Original input length: {input_length}")
        if isinstance(sequences, torch.Tensor):
            print(f"  Final sequence length: {sequences.shape[1]}")

    # Final cleanup
    del sequences
    _cleanup_gpu_memory()

    return activations, raw_outputs, input_length


def get_batch_res_activations(
    model,
    tokenizer,
    outputs,
    verbose=False,
    max_length=512,
    layers_str="auto",
):
    """
    Get activations layers for existing conversation outputs.
    Captures activations for the FULL sequence (input + generated tokens) without generation.

    Args:
        model: The language model
        tokenizer: Model tokenizer
        outputs: String or list of strings representing complete conversations/outputs
        verbose: Whether to print debug information
        max_length: Maximum sequence length for tokenization
        layers_str: String specifying which layers to extract (default: "auto" for 65% through)

    Returns:
        tuple: (activations_dict, decoded_outputs, input_length)
            - activations_dict: Dict mapping layer indices to activation tensors
            - decoded_outputs: List of conversation strings (same as input)
            - input_length: Tokenized sequence length
    """
    # Step 1: Prepare inputs - tokenize the complete outputs
    inputs = _prepare_batch_inputs(tokenizer, outputs, max_length)
    inputs = inputs.to(model.device)

    input_length = inputs.input_ids.shape[1]
    batch_size = inputs.input_ids.shape[0]

    # Use the tokenized outputs directly (no generation needed)
    sequences = inputs.input_ids

    # Clean up input tensors
    del inputs
    _cleanup_gpu_memory()

    # Step 3: Register hooks and capture activations
    hooks, activations = _register_activation_hooks(model, layers_str, verbose)

    try:
        # Forward pass to capture activations
        with torch.no_grad():
            # Build attention mask from pad token id to safely handle padding
            if isinstance(sequences, torch.Tensor):
                attention_mask = (
                    sequences.ne(tokenizer.pad_token_id).long().to(sequences.device)
                )
                with torch.no_grad():
                    model(input_ids=sequences, attention_mask=attention_mask)
                
                # Want to discard activations for the padding tokens
                for layer, acts in activations.items():
                    batch_list = []
                    for b in range(acts.size(0)):
                        valid_len = attention_mask[b].sum().item()  # number of real tokens
                        batch_list.append(acts[b, -valid_len:])     # left-pad => take last valid_len
                    activations[layer] = batch_list
                
            else:
                # Fallback for unexpected type (should not happen)
                with torch.no_grad():
                    model(sequences)
    except Exception as e:
        print(f"Error in forward pass: {e}")
    finally:
        # Always clean up hooks
        for hook in hooks:
            hook.remove()

    # Step 4: Decode outputs
    raw_outputs = _decode_outputs(tokenizer, sequences, verbose)

    if verbose:
        print("\nBatch processing summary:")
        print(f"  Processed {batch_size} prompts")
        print(f"  Original input length: {input_length}")
        if isinstance(sequences, torch.Tensor):
            print(f"  Final sequence length: {sequences.shape[1]}")

    # Final cleanup
    del sequences
    _cleanup_gpu_memory()

    return activations, raw_outputs, input_length


# * DATASET UTILS


def _sanitize_for_path(text: str) -> str:
    """Sanitize arbitrary text for safe filesystem paths."""
    text = text.replace(os.sep, "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "unnamed"


def _build_output_path(
    base_out: str,
    model_name: str,
    policy: str,
    behaviour: str,
    default_ext: str = ".jsonl",
) -> str:
    """Construct the final output path under data/<behaviour>/.

    Appends policy to the provided base filename to avoid accidental overrides.

    Args:
        base_out: Base output filename
        model_name: Model name
        policy: Policy name
        behaviour: Behaviour name
        default_ext: Default file extension if none provided (default: ".jsonl")
    """
    # safe_model = _sanitize_for_path(model_name)
    # safe_policy = _sanitize_for_path(policy)
    safe_behaviour = _sanitize_for_path(behaviour)
    base_dir = os.path.join("../data", safe_behaviour)
    os.makedirs(base_dir, exist_ok=True)

    root, ext = os.path.splitext(os.path.basename(base_out))
    ext = ext if ext else default_ext
    final_name = f"{_sanitize_for_path(root)}{ext}"
    return os.path.join(base_dir, final_name)


def _build_simple_output_path(
    base_out: str,
    model_name: str = None,
    default_ext: str = ".jsonl",
) -> str:
    """Construct a simple output path in the same directory as the base output file.

    Args:
        base_out: Base output filename (absolute or relative path)
        model_name: Model name (not used in simple mode)
        default_ext: Default file extension if none provided (default: ".jsonl")
    """
    # Simply return the base_out path as-is
    # Ensure the directory exists
    output_dir = os.path.dirname(base_out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Add default extension if none provided
    root, ext = os.path.splitext(base_out)
    if not ext:
        return f"{base_out}{default_ext}"
    return base_out


def _load_existing_results(output_file):
    """Load existing results from file if available."""
    if not os.path.exists(output_file):
        return [], 0

    try:
        if output_file.endswith(".pkl"):
            with open(output_file, "rb") as f:
                loaded_obj = pickle.load(f)
        else:
            # Load from JSON lines format
            loaded_obj = []
            with open(output_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        loaded_obj.append(json.loads(line))

        # If file contains a DataFrame (from an older/partial run), convert to list-of-dicts
        if isinstance(loaded_obj, pd.DataFrame):
            df = loaded_obj
            results = []
            for _, row in df.iterrows():
                results.append(
                    {
                        "original_index": None,
                        "input": row.get("input")
                        if "input" in df.columns
                        else row.get("input_filtered"),
                        "input_formatted": row.get("input_formatted")
                        if "input_formatted" in df.columns
                        else row.get("input"),
                        "activations": row.get("activations", {}),
                        "model_outputs": row.get("model_outputs")
                        if "model_outputs" in df.columns
                        else row.get("model_output", ""),
                    }
                )
            return results, len(results)

        # If already a list, return as-is
        if isinstance(loaded_obj, list):
            return loaded_obj, len(loaded_obj)

        print(
            f"Warning: Unexpected object type in {output_file}: {type(loaded_obj)}. Starting fresh."
        )
        return [], 0
    except (pickle.PickleError, FileNotFoundError, EOFError) as e:
        print(f"Could not load existing results: {e}. Starting fresh.")
        return [], 0


def _create_result_entry(
    original_idx, original_input, formatted_input, activations, model_output
):
    """Create a standardized result entry."""
    return {
        "original_index": original_idx,
        "input": original_input,  # Human input / raw prompt
        "input_formatted": formatted_input,  # Exact input after chat template
        "activations": activations,
        "model_outputs": model_output,
    }


def _extract_assistant_response(full_output):
    """Extract just the assistant response from a full model output sequence."""
    # Look for the assistant response after "assistant\n\n" pattern
    if "assistant\n\n" in full_output:
        # Split on "assistant\n\n" and take the last part
        parts = full_output.split("assistant\n\n")
        if len(parts) > 1:
            return parts[-1].strip()

    # Fallback: look for other patterns or return as-is
    # If no clear pattern found, return the full output
    return full_output.strip()


def _create_output_only_result_entry(
    original_idx, original_input, formatted_input, model_output
):
    """Create a result entry for output-only processing (no activations)."""
    import json

    # Extract just the assistant response (in case model_output contains full sequence)
    clean_assistant_response = _extract_assistant_response(model_output)

    # Create the conversation format with user and assistant roles
    conversation = [
        {"role": "user", "content": original_input},
        {"role": "assistant", "content": clean_assistant_response},
    ]

    return {
        "inputs": json.dumps(conversation),  # JSON string of the conversation
        "ids": f"generated_{original_idx}",  # Unique identifier
        "original_index": original_idx,  # Keep for backwards compatibility
        "input": original_input,  # Human input / raw prompt
        "input_formatted": formatted_input,  # Exact input after chat template
        "model_outputs": model_output,  # Keep for backwards compatibility (full output)
    }


def _extract_example_activations(batch_activations, example_idx):
    """Extract activations for a specific example from batch activations."""
    example_activations = {}
    for layer_idx, layer_acts in batch_activations.items():
        # layer_acts shape: [batch_size, seq_len, hidden_dim]
        example_activations[layer_idx] = layer_acts[example_idx].cpu()
    return example_activations


def _process_successful_batch(
    batch_activations, batch_outputs, batch_prompts, human_inputs, start_idx
):
    """Process a successful batch and create result entries."""
    batch_results = []
    batch_size_actual = len(batch_prompts)

    for example_idx in range(batch_size_actual):
        example_activations = _extract_example_activations(
            batch_activations, example_idx
        )

        result_entry = _create_result_entry(
            original_idx=start_idx + example_idx,
            original_input=human_inputs[start_idx + example_idx],
            formatted_input=batch_prompts[example_idx],
            activations=example_activations,
            model_output=batch_outputs[example_idx],
        )
        batch_results.append(result_entry)

    return batch_results


def _process_successful_batch_outputs_only(
    batch_outputs, batch_prompts, human_inputs, start_idx
):
    """Process a successful batch and create result entries (output-only, no activations)."""
    batch_results = []
    batch_size_actual = len(batch_prompts)

    for example_idx in range(batch_size_actual):
        result_entry = _create_output_only_result_entry(
            original_idx=start_idx + example_idx,
            original_input=human_inputs[start_idx + example_idx],
            formatted_input=batch_prompts[example_idx],
            model_output=batch_outputs[example_idx],
        )
        batch_results.append(result_entry)

    return batch_results


def _process_failed_batch(batch_prompts, human_inputs, start_idx):
    """Process a failed batch and create empty result entries."""
    batch_results = []
    batch_size_actual = len(batch_prompts)

    for i in range(batch_size_actual):
        result_entry = _create_result_entry(
            original_idx=start_idx + i,
            original_input=human_inputs[start_idx + i],
            formatted_input=batch_prompts[i],
            activations={},  # Empty dict for failed examples
            model_output="",  # Empty string for failed outputs
        )
        batch_results.append(result_entry)

    return batch_results


def _process_failed_batch_outputs_only(batch_prompts, human_inputs, start_idx):
    """Process a failed batch and create empty result entries (output-only, no activations)."""
    batch_results = []
    batch_size_actual = len(batch_prompts)

    for i in range(batch_size_actual):
        result_entry = _create_output_only_result_entry(
            original_idx=start_idx + i,
            original_input=human_inputs[start_idx + i],
            formatted_input=batch_prompts[i],
            model_output="",  # Empty string for failed outputs
        )
        batch_results.append(result_entry)

    return batch_results


def _save_results_to_file(results, output_file):
    """Save results to file atomically to avoid corruption."""
    tmp_path = f"{output_file}.tmp"
    if output_file.endswith(".pkl"):
        with open(tmp_path, "wb") as f:
            pickle.dump(results, f)
            f.flush()
            os.fsync(f.fileno())
    else:
        # Save as JSON lines format
        with open(tmp_path, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
            f.flush()
            os.fsync(f.fileno())

    os.replace(tmp_path, output_file)


def _save_dataframe_atomic(df, output_file):
    """Save a DataFrame to pickle atomically."""
    tmp_path = f"{output_file}.tmp"

    if output_file.endswith(".pkl"):
        df.to_pickle(tmp_path)
    else:
        df.to_json(tmp_path, orient="records", lines=True)

    # Ensure data hits disk
    with open(tmp_path, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp_path, output_file)


def load_incremental_results(output_file="activations_incremental.pkl"):
    """
    Load the incrementally saved results

    Returns:
        list: List of result dictionaries
    """
    if output_file.endswith(".pkl"):
        with open(output_file, "rb") as f:
            results = pickle.load(f)
    else:
        # Load from JSON lines format
        results = []
        with open(output_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    results.append(json.loads(line))
    return results


def convert_to_dataframe(results):
    """
    Convert incremental results back to DataFrame format with proper column structure

    Args:
        results: List of result dictionaries from load_incremental_results()

    Returns:
        pd.DataFrame: DataFrame with columns (input_filtered, input, model_output, activations)
    """
    # Accept both old and new keys to support resuming mid-run, but output legacy schema
    df_data = {
        "input_filtered": [
            (
                r.get("input_filtered")
                if r.get("input_filtered") is not None
                else r.get("input")
            )
            for r in results
        ],
        "input": [
            (
                r.get("input_formatted")
                if r.get("input_formatted") is not None
                else r.get("input")
            )
            for r in results
        ],
        "model_output": [
            (
                r.get("model_output")
                if r.get("model_output") is not None
                else r.get("model_outputs")
            )
            for r in results
        ],
        "activations": [r.get("activations") for r in results],
    }

    return pd.DataFrame(df_data)


def save_results_as_dataframe(
    output_file="activations_incremental.pkl", df_output_file=None
):
    """
    Load incremental results and save as a properly structured DataFrame

    Args:
        output_file: Path to the incremental results pickle file
        df_output_file: Path to save the DataFrame (defaults to adding '_dataframe' to output_file)

    Returns:
        pd.DataFrame: The final DataFrame with proper structure
    """
    if df_output_file is None:
        base_name = output_file.replace(".pkl", "")
        df_output_file = f"{base_name}_dataframe.pkl"

    # Load results and convert to DataFrame
    results = load_incremental_results(output_file)
    final_df = convert_to_dataframe(results)

    # Save the DataFrame
    final_df.to_pickle(df_output_file)
    print(f"DataFrame saved to: {df_output_file}")
    # print(f"DataFrame shape: {final_df.shape}")
    # print(f"Columns: {list(final_df.columns)}")

    return final_df


def load_jsonl_data(file_path):
    """Load data from a JSONL file and extract human/assistant content."""
    human_list = []
    assistant_list = []
    full_list = []

    try:
        with open(file_path, "r") as file:
            for line in file:
                data = json.loads(line)
                inputs = json.loads(data["inputs"])

                human = inputs[0]["content"]
                assistant = inputs[1]["content"]
                human_list.append(human)
                assistant_list.append(assistant)
                full_list.append(human + " " + assistant)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error loading data from {file_path}: {e}")
        return [], [], []

    return human_list, assistant_list, full_list


# * process batched dataframe to extract activations
def process_batched_dataframe_outputs_only(
    model,
    tokenizer,
    df,
    prompt_column,
    batch_size=1,
    output_file="outputs_incremental.pkl",
    human_input_column=None,
    do_generation=True,
    save_increment=-1,
):
    """
    Process a pandas DataFrame and save outputs incrementally after each batch (no activations).

    Args:
        model: The model to get outputs from
        tokenizer: Tokenizer for the model
        df: pandas DataFrame containing the prompts
        prompt_column: Name of the column containing the formatted prompt strings
        batch_size: Number of examples to process at once
        output_file: File to save results incrementally
        human_input_column: Name of the column containing the original human input

    Returns:
        int: Number of processed examples
    """
    # Extract raw prompts (will be formatted exactly once below)
    dataset = df[prompt_column].tolist()

    # Extract human inputs if specified
    if human_input_column and human_input_column in df.columns:
        human_inputs = df[human_input_column].tolist()
    else:
        # Fallback to using the original dataset as human input
        human_inputs = dataset

    # print("Formatting prompts...")
    if do_generation:
        # For on-policy, format as user-only prompts with generation prompt
        formatted_prompts = format_prompts_from_strings(tokenizer, dataset)
    else:
        # For off-policy, prompts are already full sequences in df[prompt_column]
        formatted_prompts = dataset

    # Calculate processing parameters
    num_batches = (len(formatted_prompts) + batch_size - 1) // batch_size
    print(
        f"Processing {len(formatted_prompts)} examples in {num_batches} batches of size {batch_size}"
    )

    # Define incremental checkpoint path (list of dicts)
    if output_file.endswith(".pkl"):
        incremental_path = output_file.replace(".pkl", "_incremental.pkl")
    elif output_file.endswith(".jsonl"):
        incremental_path = output_file.replace(".jsonl", "_incremental.jsonl")
    elif output_file.endswith(".json"):
        incremental_path = output_file.replace(".json", "_incremental.json")
    else:
        # Fallback: append _incremental before the extension or at the end
        if "." in output_file:
            name, ext = output_file.rsplit(".", 1)
            incremental_path = f"{name}_incremental.{ext}"
        else:
            incremental_path = f"{output_file}_incremental"

    # Load existing results if available (resume from incremental list file)
    results, num_existing = _load_existing_results(incremental_path)
    start_batch = num_existing // batch_size

    if num_existing > 0:
        print(
            f"Resuming from batch {start_batch} (already processed {num_existing} examples)"
        )

    # Process batches
    for batch_idx in tqdm(range(start_batch, num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(formatted_prompts))
        batch_prompts = formatted_prompts[start_idx:end_idx]

        try:
            # Clear GPU memory and get outputs only
            _cleanup_gpu_memory()
            batch_outputs, input_length = get_batch_outputs_only(
                model,
                tokenizer,
                batch_prompts,
                verbose=False,
                do_generation=do_generation,
            )

            # Process successful batch (no activations)
            batch_results = _process_successful_batch_outputs_only(
                batch_outputs, batch_prompts, human_inputs, start_idx
            )

            # Clean up GPU memory
            _cleanup_gpu_memory()
            gc.collect()

        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            # Process failed batch
            batch_results = _process_failed_batch_outputs_only(
                batch_prompts, human_inputs, start_idx
            )

        # Add batch results and save incremental checkpoints separately
        results.extend(batch_results)

        if save_increment > 0 and batch_idx % save_increment == 0:
            _save_results_to_file(results, incremental_path)

            # Also save a partial DataFrame table to the final --out path
            # Handle backwards compatibility for old results without new keys
            inputs_list = []
            ids_list = []
            for i, r in enumerate(results):
                if "inputs" in r:
                    inputs_list.append(r["inputs"])
                    ids_list.append(r["ids"])
                else:
                    # Create new format for old results
                    import json

                    # Extract just the assistant response from the full model output
                    clean_assistant_response = _extract_assistant_response(
                        r["model_outputs"]
                    )
                    conversation = [
                        {"role": "user", "content": r["input"]},
                        {"role": "assistant", "content": clean_assistant_response},
                    ]
                    inputs_list.append(json.dumps(conversation))
                    ids_list.append(f"generated_{r.get('original_index', i)}")

            partial_df = pd.DataFrame(
                {
                    "inputs": inputs_list,  # JSON string of conversation
                    "ids": ids_list,  # Unique identifiers
                    "input_formatted": [
                        r["input_formatted"] for r in results
                    ],  # Keep for backwards compatibility
                    "input": [
                        r["input"] for r in results
                    ],  # Keep for backwards compatibility
                    "model_outputs": [
                        r["model_outputs"] for r in results
                    ],  # Keep for backwards compatibility
                }
            )
            _save_dataframe_atomic(partial_df, output_file)
            # print(
            #     f"Saved batch {batch_idx + 1}/{num_batches} - Total processed: {len(results)} examples"
            # )

    # print(f"Completed processing {len(results)} examples")
    if save_increment > 0:
        print(f"Incremental checkpoints saved to: {incremental_path}")

    # Build and save final DataFrame with schema (no activations column)
    # Handle backwards compatibility for old results without new keys
    inputs_list = []
    ids_list = []
    for i, r in enumerate(results):
        if "inputs" in r:
            inputs_list.append(r["inputs"])
            ids_list.append(r["ids"])
        else:
            # Create new format for old results
            import json

            # Extract just the assistant response from the full model output
            clean_assistant_response = _extract_assistant_response(r["model_outputs"])
            conversation = [
                {"role": "user", "content": r["input"]},
                {"role": "assistant", "content": clean_assistant_response},
            ]
            inputs_list.append(json.dumps(conversation))
            ids_list.append(f"generated_{r.get('original_index', i)}")

    final_df = pd.DataFrame(
        {
            "inputs": inputs_list,  # JSON string of conversation
            "ids": ids_list,  # Unique identifiers
            "input_formatted": [
                r["input_formatted"] for r in results
            ],  # Keep for backwards compatibility
            "input": [r["input"] for r in results],  # Keep for backwards compatibility
            "model_outputs": [
                r["model_outputs"] for r in results
            ],  # Keep for backwards compatibility
        }
    )

    if output_file.endswith(".pkl"):
        final_df.to_pickle(output_file)
    else:
        final_df.to_json(output_file, orient="records", lines=True)
    print(f"Final DataFrame saved to: {output_file}")
    print(f"DataFrame shape: {final_df.shape}")
    return len(results)


def process_batched_dataframe_incremental(
    model,
    tokenizer,
    df,
    prompt_column,
    batch_size=1,
    output_file="activations_incremental.pkl",
    human_input_column=None,
    layers_str="auto",
    save_increment=-1,
):
    """
    Process a pandas DataFrame and save activations incrementally after each batch.
    Expects formatted conversation strings in prompt_column and extracts activations from them.

    Args:
        model: The model to get activations from
        tokenizer: Tokenizer for the model
        df: pandas DataFrame containing the prompts
        prompt_column: Name of the column containing the formatted conversation strings
        batch_size: Number of examples to process at once
        output_file: File to save results incrementally
        human_input_column: Name of the column containing the original human input
        layers_str: String specifying which layers to extract (default: "auto" for 65% through)

    Returns:
        int: Number of processed examples
    """
    # Extract formatted conversation strings (no additional formatting needed)
    formatted_prompts = df[prompt_column].tolist()

    # Extract human inputs if specified
    if human_input_column and human_input_column in df.columns:
        human_inputs = df[human_input_column].tolist()
    else:
        # Fallback to using the formatted prompts
        human_inputs = formatted_prompts

    # Calculate processing parameters
    num_batches = (len(formatted_prompts) + batch_size - 1) // batch_size
    print(
        f"Processing {len(formatted_prompts)} examples in {num_batches} batches of size {batch_size}"
    )

    # Define incremental checkpoint path (list of dicts)
    if output_file.endswith(".pkl"):
        incremental_path = output_file.replace(".pkl", "_incremental.pkl")
    elif output_file.endswith(".jsonl"):
        incremental_path = output_file.replace(".jsonl", "_incremental.jsonl")
    elif output_file.endswith(".json"):
        incremental_path = output_file.replace(".json", "_incremental.json")
    else:
        # Fallback: append _incremental before the extension or at the end
        if "." in output_file:
            name, ext = output_file.rsplit(".", 1)
            incremental_path = f"{name}_incremental.{ext}"
        else:
            incremental_path = f"{output_file}_incremental"

    # Load existing results if available (resume from incremental list file)
    results, num_existing = _load_existing_results(incremental_path)
    start_batch = num_existing // batch_size

    if num_existing > 0:
        print(
            f"Resuming from batch {start_batch} (already processed {num_existing} examples)"
        )

    # Process batches
    for batch_idx in tqdm(range(start_batch, num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(formatted_prompts))
        batch_prompts = formatted_prompts[start_idx:end_idx]

        try:
            # Clear GPU memory and get activations
            _cleanup_gpu_memory()
            batch_activations, batch_outputs, input_length = get_batch_res_activations(
                model,
                tokenizer,
                batch_prompts,
                verbose=False,
                layers_str=layers_str,
            )

            # Process successful batch
            batch_results = _process_successful_batch(
                batch_activations, batch_outputs, batch_prompts, human_inputs, start_idx
            )

            # Clean up GPU memory
            del batch_activations
            _cleanup_gpu_memory()
            gc.collect()

        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            # Process failed batch
            batch_results = _process_failed_batch(
                batch_prompts, human_inputs, start_idx
            )

        # Add batch results and save incremental checkpoints separately
        results.extend(batch_results)

        if save_increment > 0 and batch_idx % save_increment == 0:
            _save_results_to_file(results, incremental_path)

            # Also save a partial DataFrame table to the final --out path
            partial_df = pd.DataFrame(
                {
                    "input_formatted": [r["input_formatted"] for r in results],
                    "input": [r["input"] for r in results],
                    "model_outputs": [r["model_outputs"] for r in results],
                    "activations": [r["activations"] for r in results],
                }
            )
            _save_dataframe_atomic(partial_df, output_file)
            # print(
            #     f"Saved batch {batch_idx + 1}/{num_batches} - Total processed: {len(results)} examples"
            # )

    # print(f"Completed processing {len(results)} examples")
    if save_increment > 0:
        print(f"Incremental checkpoints saved to: {incremental_path}")

    # # Can save all layers in one dataframe
    # final_df = pd.DataFrame(
    #     {
    #         "input_formatted": [r["input_formatted"] for r in results],
    #         "input": [r["input"] for r in results],
    #         "model_outputs": [r["model_outputs"] for r in results],
    #         "activations": [r["activations"] for r in results],
    #     }
    # )
    # final_df.to_pickle(output_file)
    # print(f"Final DataFrame saved to: {output_file}")

    # Can save each layer in a separate dataframe
    keys = results[0]["activations"].keys()
    for key in tqdm(keys, desc="Saving layers"):
        # Build DataFrame directly from results
        df_split = pd.DataFrame({
            "input_formatted": [r["input_formatted"] for r in results],
            "input": [r["input"] for r in results],
            "model_outputs": [r["model_outputs"] for r in results],
            "activations": [r["activations"][key].clone().detach() for r in results],
        })
        layer_output_file = output_file.replace(".pkl", f"_layer_{key}.pkl")
        joblib.dump(df_split, layer_output_file)
        
    # print(f"DataFrame shape: {final_df.shape}")
    return len(results)


def process_file(
    model,
    tokenizer,
    dataset_path: str,
    output_file: str,
    batch_size: int = 1,
    sample: int = 0,
    layers_str: str = "auto",
    save_increment: int = -1,
):
    """Process data from a JSONL file and save results."""
    # print("\n=== File Processing ===")

    # Load data from file
    human_list, assistant_list, full_list = load_jsonl_data(dataset_path)

    # Optional sampling for quick tests
    if sample and sample > 0:
        human_list = human_list[:sample]
        assistant_list = assistant_list[:sample]
        full_list = full_list[:sample]

    if not human_list:
        print(f"No data loaded from {dataset_path}")
        return

    # print(f"Loaded {len(human_list)} examples")

    # For all policies, we format the complete conversations and extract activations
    formatted_pairs = format_prompts_from_pairs(tokenizer, human_list, assistant_list)
    df = pd.DataFrame(
        {
            "full_dialogue": formatted_pairs,
            "human_inputs": human_list,
        }
    )
    prompt_column = "full_dialogue"
    human_input_column = "human_inputs"

    # Process incrementally
    # Use simple output path (same directory as input)
    final_out_path = _build_simple_output_path(
        output_file, model.config._name_or_path, default_ext=".pkl"
    )
    # print(f"Processing data and saving to {final_out_path}...")

    num_processed = process_batched_dataframe_incremental(
        model,
        tokenizer,
        df,
        prompt_column,
        batch_size=batch_size,
        output_file=final_out_path,
        human_input_column=human_input_column,
        layers_str=layers_str,
        save_increment=save_increment,
    )
    # print(f"Processed {num_processed} examples")


def process_file_outputs_only(
    model,
    tokenizer,
    dataset_path: str,
    output_file: str,
    batch_size: int = 1,
    policy: str = "on_policy",
    behaviour: str = "refusal",
    sample: int = 0,
    extra_prompt: str = "",
    save_increment: int = -1,
):
    """Process data from a JSONL file and save outputs only (no activations).

    Args:
        extra_prompt: Additional prompt to prepend to human inputs when policy is "off_policy_prompt"

    policy options:
      - on_policy: feed human prompts and generate model outputs
      - off_policy_prompt: feed human prompts with extra_prompt prepended and generate model outputs
      - off_policy_other_model: feed human+assistant as fixed targets (no generation)
    """
    # print("\n=== File Processing (Outputs Only) ===")

    # Load data from file
    human_list, assistant_list, full_list = load_jsonl_data(dataset_path)
    # print("json loaded!")

    # Optional sampling for quick tests
    if sample and sample > 0:
        human_list = human_list[:sample]
        assistant_list = assistant_list[:sample]
        full_list = full_list[:sample]

    if not human_list:
        print(f"No data loaded from {dataset_path}")
        return

    # print(f"Loaded {len(human_list)} examples")

    # Build DataFrame and controls based on policy
    if policy == "on_policy":
        df = pd.DataFrame({"human_inputs": human_list})
        prompt_column = "human_inputs"
        do_generation = True
        human_input_column = "human_inputs"
    else:
        # For off_policy_prompt, prepend extra_prompt to human inputs and generate new responses
        if policy == "off_policy_prompt":
            modified_human_list = [f"{extra_prompt} {human}" for human in human_list]
            df = pd.DataFrame(
                {
                    "human_inputs": modified_human_list,  # Modified inputs for generation
                    "original_human_inputs": human_list,  # Original inputs for "inputs" field
                }
            )
            prompt_column = "human_inputs"
            do_generation = True  # Generate new responses based on modified inputs
            human_input_column = (
                "original_human_inputs"  # Use original for "inputs" field
            )
        else:
            raise ValueError(
                "unknown policy"
                "Please provide an --extra-prompt argument if policy == off_policy_prompt."
            )

    # Process incrementally
    # Build deterministic output path under data/<behaviour>/
    final_out_path = _build_output_path(
        output_file, model.config._name_or_path, policy, behaviour
    )
    # print(f"Processing data and saving to {final_out_path}...")

    num_processed = process_batched_dataframe_outputs_only(
        model,
        tokenizer,
        df,
        prompt_column,
        batch_size=batch_size,
        output_file=final_out_path,
        human_input_column=human_input_column,
        do_generation=do_generation,
        save_increment=save_increment,
    )

    # print(f"Processed {num_processed} examples")
