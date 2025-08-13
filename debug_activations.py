#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gen_data.utils import (
    format_prompts_from_pairs,
    get_batch_res_activations,
    get_model,
    load_jsonl_data,
)

# Load model
print("Loading model...")
model, tokenizer = get_model("meta-llama/Llama-3.2-3B-Instruct")

# Load sample data
print("Loading data...")
human_list, assistant_list, full_list = load_jsonl_data(
    "data/refusal/off_policy_raw.jsonl"
)

# Take just 1 example for testing
human_list = human_list[:1]
assistant_list = assistant_list[:1]

print(f"Human: {human_list[0]}")
print(f"Assistant: {assistant_list[0]}")

# Format the conversation
print("Formatting conversation...")
formatted_pairs = format_prompts_from_pairs(tokenizer, human_list, assistant_list)
print(f"Formatted conversation length: {len(formatted_pairs[0])}")
print(f"Formatted conversation preview: {formatted_pairs[0][:200]}...")

# Extract activations
print("Extracting activations...")
try:
    batch_activations, batch_outputs, input_length = get_batch_res_activations(
        model,
        tokenizer,
        formatted_pairs,
        verbose=True,
    )

    print(f"Activation keys: {list(batch_activations.keys())}")
    print(f"Number of activation layers: {len(batch_activations)}")

    if len(batch_activations) > 0:
        first_key = list(batch_activations.keys())[0]
        print(
            f"Shape of first activation layer {first_key}: {batch_activations[first_key].shape}"
        )

    print(f"Batch outputs: {batch_outputs}")
    print(f"Input length: {input_length}")

except Exception as e:
    print(f"Error during activation extraction: {e}")
    import traceback

    traceback.print_exc()
