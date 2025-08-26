# Standard library imports
import argparse
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
from huggingface_hub import login

from probe_gen.gen_data.utils import get_model, process_file_outputs_only, _build_output_path
from probe_gen.config import MODELS, LABELLING_SYSTEM_PROMPTS

from probe_gen.annotation.interface_dataset import Dataset, LabelledDataset
from probe_gen.annotation.label_dataset import label_and_save_dataset



hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

def yes_no_str(v):
    v = v.lower()
    if v in ("yes", "y", "true", "t", "1"):
        return True
    elif v in ("no", "n", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Expected yes or no")


def main():
    """CLI entrypoint for output generation without activation extraction."""
    parser = argparse.ArgumentParser(description="Output generation (no activations)")
    parser.add_argument("--model", default="llama_3b")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--data", default="data/refusal/anthropic_raw_apr_23.jsonl")
    parser.add_argument(
        "--out-path",
        type=str,
        default="data/refusal/out.jsonl",
        help="Output directory for the dataset",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample", type=int, default=0, help="If >0, run on N samples")
    parser.add_argument(
        "--behaviour",
        type=str,
        default="refusal",
        help="Name of the behaviour bucket; outputs saved under datasets/<behaviour>/",
    )
    parser.add_argument(
        "--add-prompt",
        type=str,
        default="no",
        help="Whether to add extra prompt to human inputs for prompted behaviour (e.g., 'refuse this answer')",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="alternating",
        help="When to use only positive prompts ('positive'), only negative ('negative'), or alternate ('alternate')",
    )
    parser.add_argument(
        "--save-increment",
        type=int,
        default=-1,
        help="Number of batches between each checkpoint, set to -1 to never save",
    )
    parser.add_argument(
        "--num-balanced",
        type=int,
        default=5000,
        help="How large the balanced dataset should be, stopping labelling early if reached",
    )
    args = parser.parse_args()

    ########## OUTPUTS ###########

    print(f"Loading model: {MODELS[args.model]}")
    model, tokenizer = get_model(MODELS[args.model])

    process_file_outputs_only(
        model,
        tokenizer,
        temperature=args.temperature,
        dataset_path=args.data,
        output_file="temp_outputs",
        batch_size=args.batch_size,
        behaviour=args.behaviour,
        sample=args.sample,
        add_prompt=yes_no_str(args.add_prompt),
        prompt_type=args.prompt_type,
        save_increment=args.save_increment,
    )

    ########## LABELS ###########

    system_prompt = LABELLING_SYSTEM_PROMPTS[args.behaviour]

    # Load the dataset
    in_path = _build_output_path("temp_outputs", args.behaviour)
    try:
        dataset = LabelledDataset.load_from(in_path)
    except Exception:
        dataset = Dataset.load_from(in_path)

    label_and_save_dataset(
        dataset=dataset,
        dataset_path=args.out_path,
        system_prompt=system_prompt,
        do_subsample=True,
        num_balanced=args.num_balanced,
    )


if __name__ == "__main__":
    main()
