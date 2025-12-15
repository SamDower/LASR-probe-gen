# Standard library imports
import os
import sys
from pathlib import Path

from probe_gen.annotation.datasets import *
from probe_gen.config import MODELS
from probe_gen.gen_data.utils import get_model, process_file

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import os
import shutil

from huggingface_hub import HfApi, hf_hub_download, login

# os.environ["HF_TOKEN"] = ""
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    raise ValueError("HF_TOKEN is not set")

def wipe_directory(dir_path: Path):
    """Delete all files and subdirectories inside dir_path, but keep dir_path itself."""
    if not dir_path.exists():
        return  # nothing to do
    
    for item in dir_path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()  # remove file or symlink
        elif item.is_dir():
            shutil.rmtree(item)  # remove directory recursively

def generate_and_save_activations(behaviour, datasource, activations_model, response_model, generation_method, mode, layers, balanced_responses_filepath, activations_batch_size):
    print(f"Loading model: {MODELS[activations_model]}")
    model, tokenizer = get_model(MODELS[activations_model])

    # Generate output filename automatically in the same directory as input
    input_dir = os.path.dirname(balanced_responses_filepath)
    input_basename = os.path.splitext(os.path.basename(balanced_responses_filepath))[0]
    output_file = os.path.join(input_dir, f"{input_basename}.pkl")

    layers_str = ",".join(map(str, layers))

    process_file(
        model,
        tokenizer,
        dataset_path=balanced_responses_filepath,
        output_file=output_file,
        batch_size=activations_batch_size,
        sample=0,
        layers_str=layers_str,
        save_increment=-1,
        include_prompt="included" in generation_method
    )

    REPO_NAME = f"lasrprobegen/{behaviour}-activations"
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    api = HfApi()
    print("Creating repository...")
    api.create_repo(
        repo_id=REPO_NAME,
        repo_type="dataset",
        token=hf_token,
        private=False,
        exist_ok=True
    )

    for layer in layers:
        file_path = str(output_file).replace(".pkl", f"_layer_{layer}.pkl")
        path_in_repo = f"{datasource}/{activations_model}/{response_model}_{generation_method}_{mode}_layer_{layer}.pkl"
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=path_in_repo,
            repo_id=REPO_NAME,
            repo_type="dataset",
            token=hf_token,
        )


def main():
    # ====== All settings to change are here ======
    response_model = "ministral_8b"
    activations_model = "llama_3b"
    generation_methods = ["on_policy"]
    layers = [0,3,6,9,12,15,18,21,24,27]
    experiments = [
        # {"behaviour": "authority", "datasources": ["multichoice", "arguments"]},
        {"behaviour": "bias", "datasources": ["arguments"]},
        # {"behaviour": "lists", "datasources": ["ultrachat", "shakespeare"]},
        # {"behaviour": "metaphors", "datasources": ["ultrachat", "writingprompts"]},
        # {"behaviour": "science", "datasources": ["ultrachat", "mmlu"]},
        # {"behaviour": "refusal", "datasources": ["jailbreaks"]},
    ]
    activations_batch_size = 32
    train_test_modes = ["train", "test"]
    # ==============================================

    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Loop over each experiment
    for exp in experiments:
        try:
            behaviour = exp["behaviour"]
            datasources = exp["datasources"]

            for datasource in datasources:
                for generation_method in generation_methods:
                    for mode in train_test_modes:
                        try:
                            repo_id = f"lasrprobegen/{behaviour}-activations"
                            local_dir = "data/temp"
                            os.makedirs(local_dir, exist_ok=True)
                            generation_method_download_name = generation_method.replace("_included", "")
                            file_path = hf_hub_download(
                                repo_id=repo_id,
                                repo_type="dataset",
                                filename=f"{datasource}/{response_model}_{generation_method_download_name}_{mode}.jsonl",
                                local_dir=local_dir,
                                token=os.getenv("HF_TOKEN")
                            )
                            print(f"Downloaded to: {file_path}")
                            generate_and_save_activations(behaviour, datasource, activations_model, response_model, generation_method, mode, layers, f"data/temp/{datasource}/{response_model}_{generation_method_download_name}_{mode}.jsonl", activations_batch_size)
                            # Delete all files that were used
                            wipe_directory(temp_dir)
                        except Exception as e:
                            print(f"Error generating datasets for {behaviour} {datasource} train: {e}")
                            wipe_directory(temp_dir)
                            continue

        except Exception as e:
            print(f"Error generating datasets for {exp}: {e}")
            wipe_directory(temp_dir)
            continue

if __name__ == "__main__":
    main()