# Standard library imports
import argparse
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe_gen.annotation.interface_dataset import Dataset, LabelledDataset
from probe_gen.config import LABELLING_SYSTEM_PROMPTS
from probe_gen.labelling.label_dataset import label_and_save_dataset
from probe_gen.labelling.refusal_autograder import grade_data_harmbench


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
        "--in-path",
        type=str,
        default=None,
        help="Input directory of an existing dataset (optional)",
    )
    parser.add_argument(
        "--num-balanced",
        type=int,
        default=5000,
        help="How large the balanced dataset should be, stopping labelling early if reached",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (useful for testing)",
    )
    parser.add_argument(
        "--only-resplit",
        action="store_true",
        help="Only resplit the dataset, not grade it",
    )
    parser.add_argument(
        "--single_set_size",
       type=int,
        default=None,
        help="Only create one set of length single_set_size",
    )
    args = parser.parse_args()

    # Load the dataset
    try:
        dataset = LabelledDataset.load_from(args.in_path)
    except Exception:
        dataset = Dataset.load_from(args.in_path)

    if args.behaviour == "jailbreaks":
        grade_data_harmbench(
            filename=args.in_path,
            output_path=args.out_path,
            num_balanced=args.num_balanced,
            max_samples=args.max_samples,
            only_resplit = args.only_resplit,
            single_set_size = args.single_set_size,
        )
        
    else:
        system_prompt = LABELLING_SYSTEM_PROMPTS[args.behaviour]
        label_and_save_dataset(
            dataset=dataset,
            dataset_path=args.out_path,
            system_prompt=system_prompt,
            do_subsample=True, # TODO: investigate
            num_balanced=args.num_balanced,
        )


if __name__ == "__main__":
    main()
