from probe_gen.annotation.datasets import *

# from probe_gen.labelling.sycophancy_multichoice_autograder import (
#     label_and_save_dataset_sycophancy_multichoice,
# )
from probe_gen.labelling.arguments_autograder import label_and_save_dataset_arguments

if __name__ == "__main__":

    label_and_save_dataset_arguments(
        responses_file="data/sycophancy/arguments/llama_responses_5k.jsonl",
        out_file="data/sycophancy/arguments/qwen_labels_5k.jsonl",
        num_balanced=10000000,
        include_non_determined=True,
    )