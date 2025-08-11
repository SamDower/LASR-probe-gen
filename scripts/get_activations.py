# Standard library imports

# Third-party imports
import argparse
import os

from huggingface_hub import login

from get_activations.utils import get_model, process_file

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)


def main():
    """CLI entrypoint for activation extraction."""
    parser = argparse.ArgumentParser(description="Activation extraction")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--data", default="data/refusal/anthropic_raw_apr_23.jsonl")
    parser.add_argument("--out", default="my_activations.pkl")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample", type=int, default=0, help="If >0, run on N samples")
    parser.add_argument(
        "--policy",
        type=str,
        default="on_policy",
        choices=["on_policy", "off_policy_prompt", "off_policy_other_model"],
        help="Activation policy: generate (on_policy) or teacher-forced (off_policy_*)",
    )
    parser.add_argument(
        "--behaviour",
        type=str,
        default="refusal",
        help="Name of the behaviour bucket; outputs saved under datasets/<behaviour>/",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = get_model(args.model)

    process_file(
        model,
        tokenizer,
        dataset_path=args.data,
        output_file=args.out,
        batch_size=args.batch_size,
        policy=args.policy,
        behaviour=args.behaviour,
        sample=args.sample,
    )


if __name__ == "__main__":
    main()
