import pandas as pd
from datasets import Dataset, DatasetDict
import argparse
import os
from huggingface_hub import HfApi, upload_file

def upload_csv_to_hf(csv_path, repo_name):
    """
    Uploads a CSV file to the Hugging Face Hub.

    Args:
        csv_path (str): The path to the CSV file.
        repo_name (str): The name of the repository on the Hugging Face Hub.
    """
    api = HfApi()
    
    # Ensure the repository exists
    api.create_repo(repo_id=repo_name, repo_type="dataset", exist_ok=True)
    
    file_name = os.path.basename(csv_path)
    print(f"Uploading {file_name} to {repo_name}...")

    # Upload the file
    upload_file(
        path_or_fileobj=csv_path,
        path_in_repo=file_name,
        repo_id=repo_name,
        repo_type="dataset",
    )
    
    print("Upload complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a CSV file to a Hugging Face Hub dataset repository.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file.")
    parser.add_argument("repo_name", type=str, help="Name of the repository on the Hugging Face Hub (e.g., 'username/my-dataset').")
    args = parser.parse_args()

    upload_csv_to_hf(args.csv_path, args.repo_name)
