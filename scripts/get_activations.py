# Standard library imports
import argparse
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
from huggingface_hub import login

from probe_gen.gen_data.utils import get_model, process_file
from probe_gen.config import MODELS

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)


def main():
    """CLI entrypoint for activation extraction."""
    parser = argparse.ArgumentParser(description="Activation extraction")
    parser.add_argument("--model", default="llama_3b")
    parser.add_argument("--data", default="data/refusal/anthropic_raw_apr_23.jsonl")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample", type=int, default=0, help="If >0, run on N samples")
    parser.add_argument(
        "--layers",
        type=str,
        default="auto",
        help="Which layers to extract activations from. Options: 'all' (all layers), 'auto' (65%% through), comma-separated indices '0,5,10', ranges '5-10', or mixed '0,5-8,15'",
    )
    parser.add_argument(
        "--save-increment",
        type=int,
        default=-1,
        help="Number of batches between each checkpoint, set to -1 to never save",
    )
    args = parser.parse_args()

    print(f"Loading model: {MODELS[args.model]}")
    model, tokenizer = get_model(MODELS[args.model])

    # Generate output filename automatically in the same directory as input
    input_dir = os.path.dirname(args.data)
    input_basename = os.path.splitext(os.path.basename(args.data))[0]
    output_file = os.path.join(input_dir, f"{input_basename}.pkl")

    process_file(
        model,
        tokenizer,
        dataset_path=args.data,
        output_file=output_file,
        batch_size=args.batch_size,
        sample=args.sample,
        layers_str=args.layers,
        save_increment=args.save_increment,
    )


if __name__ == "__main__":
    main()
