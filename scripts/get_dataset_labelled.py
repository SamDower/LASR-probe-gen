# Standard library imports
import argparse
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe_gen.annotation.interface_dataset import Dataset, LabelledDataset
from probe_gen.annotation.label_dataset import label_and_save_dataset
from probe_gen.annotation.refusal_behaviour import (
    SYSTEM_PROMPT_REFUSE,
    create_refusal_dataset,
)
from probe_gen.annotation.ultrachat_behaviour import (
    SYSTEM_PROMPT_LIST,
    SYSTEM_PROMPT_FORMALITY,
    SYSTEM_PROMPT_METAPHORS,
    create_ultrachat_dataset,
)


def yes_no_str(v):
    v = v.lower()
    if v in ("yes", "y", "true", "t", "1"):
        return True
    elif v in ("no", "n", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Expected yes or no")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--behaviour",
        type=str,
        default="refusal",
        help="Which behaviour to get dataset or labels for",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="data/refusal/out.jsonl",
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to use (not used if in_path is provided)",
    )
    parser.add_argument(
        "--in-path",
        type=str,
        default=None,
        help="Input directory of an existing dataset (optional)",
    )
    parser.add_argument(
        "--do-label",
        type=yes_no_str,
        default=True,
        help="Whether to label the dataset (y/n)",
    )
    parser.add_argument(
        "--do-subsample",
        type=yes_no_str,
        default=True,
        help="Whether to subsample the dataset (y/n)",
    )
    parser.add_argument(
        "--num-balanced",
        type=int,
        default=5000,
        help="How large the balanced dataset should be, stopping labelling early if reached",
    )
    args = parser.parse_args()

    dataset = None
    if args.behaviour == "refusal":
        system_prompt = SYSTEM_PROMPT_REFUSE
        # Sample the dataset
        if args.in_path is None:
            dataset = create_refusal_dataset(num_samples=args.num_samples)

    else:  # Currently can assume behaviour must be ultrachat
        if args.behaviour == "lists":
            system_prompt = SYSTEM_PROMPT_LIST
        elif args.behaviour == "formality":
            system_prompt = SYSTEM_PROMPT_FORMALITY
        elif args.behaviour == "metaphors":
            system_prompt = SYSTEM_PROMPT_METAPHORS
        else:
            raise ValueError(f"Behaviour {args.behaviour} not supported")
        # Sample the dataset
        if args.in_path is None:
            dataset = create_ultrachat_dataset(num_samples=args.num_samples)

    # If didnt sample, load the dataset
    if dataset is None:
        try:
            dataset = LabelledDataset.load_from(args.in_path)
        except Exception:
            dataset = Dataset.load_from(args.in_path)

    label_and_save_dataset(
        dataset=dataset,
        dataset_path=args.out_path,
        system_prompt=system_prompt,
        do_subsample=args.do_subsample,
        do_label=args.do_label,
        num_balanced=args.num_balanced,
    )


if __name__ == "__main__":
    main()
