# Standard library imports
import gc
import json

# Suppress transformers warnings about unused generation parameters
import logging
import os
import pickle

# Third-party imports
# import numpy as np  # Imported but not used
import pandas as pd
import torch

logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

# Optional: Login to HuggingFace Hub if needed
# Uncomment and set your token in environment variable HF_TOKEN
import argparse

from huggingface_hub import login

# from huggingface_hub import login  # Uncomment if needed
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)


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
    print(f"Using pad token for {model_id}: {pad_token}")
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

    # Set the pad token using the same logic as control experiments
    tokenizer.pad_token = get_pad_token(model_name, tokenizer)
    print(f"Pad token set to: {tokenizer.pad_token}")

    return model, tokenizer


# Note: Model and tokenizer are now loaded in main() via CLI args


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


def _prepare_batch_inputs(tokenizer, prompts, max_length=512):
    """Prepare and tokenize batch inputs."""
    # Handle single prompt case
    if isinstance(prompts, str):
        prompts = [prompts]

    return tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )


def _generate_sequences(model, tokenizer, inputs, max_new_tokens=200, temperature=0.7):
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
        base_args = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        }

        # Smart temperature handling with clean parameter passing
        if temperature == 0.0:
            # Use greedy decoding - only pass necessary parameters
            generation_args = {
                **base_args,
                "do_sample": False,  # Greedy decoding
            }
        elif 0.0 < temperature <= 2.0:
            # Use sampling with temperature
            generation_args = {
                **base_args,
                "do_sample": True,
                "temperature": temperature,
            }
        else:
            # Temperature too high - clamp and warn
            clamped_temp = min(temperature, 2.0)
            print(
                f"Warning: Temperature {temperature} is very high. Clamping to {clamped_temp}"
            )
            generation_args = {
                **base_args,
                "do_sample": True,
                "temperature": clamped_temp,
            }

        outputs = model.generate(**generation_args)
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


def _register_activation_hooks(model, verbose=False):
    """Register hooks on all model layers and return the hooks and activations dict."""
    activations = {}
    layers_to_enum = get_res_layers_to_enumerate(model)
    hooks = []

    for i, layer in enumerate(layers_to_enum):
        hook_fn = _create_activation_hook(activations, i, verbose)
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


def _cleanup_gpu_memory():
    """Clean up GPU memory if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_batch_res_activations(
    model,
    tokenizer,
    prompts,
    verbose=False,
    max_length=512,
    max_new_tokens=200,
    temperature=0.0,
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

    # Step 2: Generate sequences
    outputs = _generate_sequences(model, tokenizer, inputs, max_new_tokens, temperature)

    # Clean up input tensors
    del inputs
    _cleanup_gpu_memory()

    # Step 3: Register hooks and capture activations
    hooks, activations = _register_activation_hooks(model, verbose)

    try:
        # Forward pass to capture activations
        with torch.no_grad():
            model(outputs)
    finally:
        # Always clean up hooks
        for hook in hooks:
            hook.remove()

    # Step 4: Decode outputs
    raw_outputs = _decode_outputs(tokenizer, outputs, verbose)

    if verbose:
        print("\nBatch processing summary:")
        print(f"  Processed {batch_size} prompts")
        print(f"  Original input length: {input_length}")
        print(f"  Final sequence length: {outputs.shape[1]}")

    # Final cleanup
    del outputs
    _cleanup_gpu_memory()

    return activations, raw_outputs, input_length


def _load_existing_results(output_file):
    """Load existing results from file if available."""
    if not os.path.exists(output_file):
        return [], 0

    try:
        with open(output_file, "rb") as f:
            loaded_obj = pickle.load(f)

        # If file contains a DataFrame (from an older run), convert to list-of-dicts
        if isinstance(loaded_obj, pd.DataFrame):
            df = loaded_obj
            # Accept both legacy and new column names
            # Expected new columns: input_formatted, input, model_outputs, activations
            # Legacy columns: input_filtered, input, model_output, activations
            results = []
            for _, row in df.iterrows():
                result_entry = {
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
                results.append(result_entry)
            return results, len(results)

        # If file already contains a list-like, return as-is
        if isinstance(loaded_obj, list):
            return loaded_obj, len(loaded_obj)

        # Fallback: unknown object type, start fresh
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


def _save_results_to_file(results, output_file):
    """Save results to pickle file atomically to avoid corruption."""
    tmp_path = f"{output_file}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(results, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_file)


def _save_dataframe_atomic(df, output_file):
    """Save a DataFrame to pickle atomically."""
    tmp_path = f"{output_file}.tmp"
    df.to_pickle(tmp_path)
    # Ensure data hits disk
    with open(tmp_path, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp_path, output_file)


def process_batched_dataframe_incremental(
    model,
    tokenizer,
    df,
    prompt_column,
    batch_size=1,
    output_file="activations_incremental.pkl",
    human_input_column=None,
):
    """
    Process a pandas DataFrame and save activations incrementally after each batch.

    Args:
        model: The model to get activations from
        tokenizer: Tokenizer for the model
        df: pandas DataFrame containing the prompts
        prompt_column: Name of the column containing the formatted prompt strings
        batch_size: Number of examples to process at once
        output_file: File to save results incrementally
        human_input_column: Name of the column containing the original human input (for input_filtered)

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

    print("Formatting prompts...")
    formatted_prompts = format_prompts_from_strings(tokenizer, dataset)

    # Calculate processing parameters
    num_batches = (len(formatted_prompts) + batch_size - 1) // batch_size
    print(
        f"Processing {len(formatted_prompts)} examples in {num_batches} batches of size {batch_size}"
    )

    # Define incremental checkpoint path (list of dicts)
    incremental_path = (
        output_file.replace(".pkl", "_incremental.pkl")
        if output_file.endswith(".pkl")
        else f"{output_file}_incremental.pkl"
    )

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
                model, tokenizer, batch_prompts, verbose=False
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

        print(
            f"Saved batch {batch_idx + 1}/{num_batches} - Total processed: {len(results)} examples"
        )

    print(f"Completed processing {len(results)} examples")
    print(f"Incremental checkpoints saved to: {incremental_path}")

    # Build and save final DataFrame with new schema directly to --out
    final_df = pd.DataFrame(
        {
            "input_formatted": [r["input_formatted"] for r in results],
            "input": [r["input"] for r in results],
            "model_outputs": [r["model_outputs"] for r in results],
            "activations": [r["activations"] for r in results],
        }
    )
    final_df.to_pickle(output_file)
    print(f"Final DataFrame saved to: {output_file}")
    print(f"DataFrame shape: {final_df.shape}")
    return len(results)


def load_incremental_results(output_file="activations_incremental.pkl"):
    """
    Load the incrementally saved results

    Returns:
        list: List of result dictionaries
    """
    with open(output_file, "rb") as f:
        results = pickle.load(f)
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
    print(f"DataFrame shape: {final_df.shape}")
    print(f"Columns: {list(final_df.columns)}")

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


def process_file(
    model, tokenizer, dataset_path: str, output_file: str, batch_size: int = 1
):
    """Process data from a JSONL file and save results."""
    print("\n=== File Processing ===")

    # Load data from file
    human_list, assistant_list, full_list = load_jsonl_data(dataset_path)

    if not human_list:
        print(f"No data loaded from {dataset_path}")
        return

    print(f"Loaded {len(human_list)} examples")

    # Create DataFrame with raw human inputs only (formatting happens once later)
    df = pd.DataFrame({"human_inputs": human_list})

    # Process incrementally
    print(f"Processing data and saving to {output_file}...")

    num_processed = process_batched_dataframe_incremental(
        model,
        tokenizer,
        df,
        "human_inputs",
        batch_size=1,
        output_file=output_file,
        human_input_column="human_inputs",
    )

    print(f"Processed {num_processed} examples")

    # Conversion/inspection removed per request


def main():
    """CLI entrypoint for activation extraction."""
    parser = argparse.ArgumentParser(description="Activation extraction")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--data", default="data/refusal/anthropic_raw_apr_23.jsonl")
    parser.add_argument("--out", default="my_activations.pkl")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample", type=int, default=0, help="If >0, run on N samples")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = get_model(args.model)

    process_file(
        model,
        tokenizer,
        dataset_path=args.data,
        output_file=args.out,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
