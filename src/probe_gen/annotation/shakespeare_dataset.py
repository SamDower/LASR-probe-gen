"""
Shakespeare dataset implementation as a PromptDataset subclass.

This module provides a clean interface for generating Shakespeare roleplay datasets
with automatic train/test separation and consistent interface. Uses the Kaggle
Shakespeare plays dataset for creating roleplay prompts.
"""

import csv
import os
from collections import defaultdict

import kagglehub
from tqdm import tqdm

from probe_gen.annotation.interface_dataset import Dataset, Message
from probe_gen.annotation.prompt_dataset import PromptDataset
from probe_gen.paths import data


class ShakespeareDataset(PromptDataset):
    """
    Shakespeare dataset implementation with unified PromptDataset interface.
    
    This class handles Shakespeare plays dataset from Kaggle with automatic
    data downloading and proper train/test separation. Creates roleplay prompts
    based on Shakespeare play lines.
    
    Features:
    - Automatic download from Kaggle (guslovesmath/shakespeare-plays-dataset)
    - Filters lines by minimum length for quality
    - Adds roleplay instruction to each line
    - Proper train/test separation using skip parameters
    - Consistent interface with other prompt datasets
    """
    
    def __init__(
        self, 
        min_prompt_length: int = 40,
        kaggle_dataset_id: str = "guslovesmath/shakespeare-plays-dataset",
        default_train_test_gap: int = 45000
    ):
        """
        Initialize Shakespeare prompt dataset.
        
        Args:
            min_prompt_length: Minimum length for Shakespeare lines to include
            kaggle_dataset_id: Kaggle dataset identifier
            default_train_test_gap: Default gap between train and test data
        """
        super().__init__("shakespeare", default_train_test_gap)
        
        self.min_prompt_length = min_prompt_length
        self.kaggle_dataset_id = kaggle_dataset_id
        self._cached_lines = None  # Cache processed lines to avoid reprocessing
        self._csv_path = None  # Store the path to the CSV file
        
        # Instruction template for roleplay
        self.instruction_template = "Roleplay as a shakespearean character briefly replying to this line: "
    
    def download_data(self) -> None:
        """
        Download Shakespeare plays dataset from Kaggle if not already available.
        
        First checks if the CSV file exists locally, otherwise downloads from Kaggle.
        """
        local_path = data.data / "shakespeare_plays.csv"
        
        if local_path.exists():
            print(f"Found existing Shakespeare dataset at {local_path}")
            self._csv_path = str(local_path)
        else:
            print(f"Downloading Shakespeare dataset from Kaggle: {self.kaggle_dataset_id}")
            try:
                # Download from Kaggle
                download_path = kagglehub.dataset_download(self.kaggle_dataset_id)
                kaggle_csv_path = os.path.join(download_path, "shakespeare_plays.csv")
                
                if not os.path.exists(kaggle_csv_path):
                    raise FileNotFoundError(f"Expected CSV file not found after download: {kaggle_csv_path}")
                
                self._csv_path = kaggle_csv_path
                print(f"✓ Successfully downloaded Shakespeare dataset to {kaggle_csv_path}")
                
            except Exception as e:
                raise RuntimeError(f"Failed to download Shakespeare dataset: {e}")
    
    def _process_and_cache_lines(self) -> None:
        """
        Process the CSV file and cache valid lines.
        
        Filters lines by length and adds roleplay instructions.
        """
        if self._cached_lines is not None:
            return  # Already cached
        
        if self._csv_path is None:
            raise ValueError("Data not downloaded. Call download_data() first.")
        
        print(f"Processing Shakespeare lines from {self._csv_path}...")
        
        processed_lines = []
        
        with open(self._csv_path, 'r', encoding='utf-8') as f:
            dataset_reader = csv.reader(f)
            
            for real_ix, sample in tqdm(enumerate(dataset_reader), desc="Processing lines"):
                # Skip empty rows
                if not sample or sample[0] == "":
                    continue
                
                # Get the line content (second to last column based on original code)
                if len(sample) < 2:
                    continue
                
                line_content = sample[-2]  # Second to last column
                
                # Filter by minimum length
                if len(line_content) < self.min_prompt_length:
                    continue
                
                # Add roleplay instruction
                prompt = self.instruction_template + line_content
                
                # Store the processed line with its original index
                processed_lines.append((prompt, real_ix))
        
        self._cached_lines = processed_lines
        print(f"✓ Processed {len(processed_lines)} valid Shakespeare lines")
    
    def _create_dataset(self, mode: str, n_samples: int, skip: int) -> Dataset:
        """
        Create Shakespeare dataset with specified parameters.
        
        Args:
            mode: "train" or "test" (used for identification only)
            n_samples: Number of samples to generate
            skip: Number of samples to skip from the beginning
            
        Returns:
            Dataset object with Shakespeare roleplay prompts
            
        Raises:
            ValueError: If not enough samples are available
        """
        # Ensure data is processed and cached
        self._process_and_cache_lines()
        
        # Check if we have enough arguments after skipping
        total_available = len(self._cached_lines)
        end_index = skip + n_samples
        if end_index > total_available:
            end_index = total_available
            print("Warning: Not enough data available, using the remainder")
        if mode == "train" and end_index > self.default_train_test_gap:
            end_index = self.default_train_test_gap
            print("Warning: Not enough data available, using the remainder")
        
        # Extract the requested slice
        selected_lines = self._cached_lines[skip:end_index]
        
        ids = []
        inputs = []
        other_fields = defaultdict(list)
        
        # Process the selected lines
        for processed_ix, (prompt, original_ix) in enumerate(selected_lines):
            # Create conversation messages with user prompt and empty assistant response
            messages = [
                Message(role="user", content=prompt),
                Message(role="assistant", content="")
            ]
            
            # Create unique sample ID
            sample_id = f"_{original_ix}_text_{mode}"
            
            ids.append(sample_id)
            inputs.append(messages)
            
            # Add metadata
            other_fields["original_index"].append(original_ix)
            other_fields["processed_index"].append(skip + processed_ix)
            other_fields["line_length"].append(len(prompt) - len(self.instruction_template))
        
        dataset = Dataset(inputs=inputs, ids=ids, other_fields=other_fields)
        return dataset
    
    def get_dataset_info(self):
        """Get information about this Shakespeare dataset."""
        info = super().get_dataset_info()
        info.update({
            'kaggle_dataset_id': self.kaggle_dataset_id,
            'min_prompt_length': self.min_prompt_length,
            'instruction_template': self.instruction_template,
            'csv_path': self._csv_path,
            'total_processed_lines': len(self._cached_lines) if self._cached_lines is not None else "Unknown (data not processed)",
            'description': "Shakespeare roleplay dataset based on play lines"
        })
        return info


# Example usage and testing
if __name__ == "__main__":
    print("Testing ShakespeareDataset...")
    
    # Test dataset initialization
    print("\n" + "="*50)
    print("Testing Shakespeare dataset...")
    
    shakespeare_dataset = ShakespeareDataset(
        min_prompt_length=30  # Lower threshold for testing
    )
    
    print(f"\nDataset info: {shakespeare_dataset.get_dataset_info()}")
    
    # Test data download (this would download from Kaggle if not present)
    try:
        print("\nTesting data download...")
        shakespeare_dataset.download_data()
        print("✓ Data download successful")
        
        # Test small dataset generation
        print("\nTesting dataset generation...")
        results = shakespeare_dataset.generate_data(
            train_samples=10,
            test_samples=5,
            save_dir="/tmp/shakespeare_test"
        )
        
        print("\n✓ Generated datasets:")
        print(f"  Train: {len(results['train'])} samples")
        print(f"  Test: {len(results['test'])} samples")
        
        # Show example prompt
        example_prompt = results['train'].inputs[0][0].content
        print(f"\nExample prompt:\n{example_prompt[:200]}...")
        
        # Show some metadata
        if results['train'].other_fields:
            print("\nExample metadata:")
            print(f"  Original index: {results['train'].other_fields['original_index'][0]}")
            print(f"  Line length: {results['train'].other_fields['line_length'][0]}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Note: This might fail if Kaggle credentials are not configured")
    
    # Test single dataset mode
    print("\n" + "="*50)
    print("Testing single dataset mode...")
    
    try:
        results = shakespeare_dataset.generate_data(
            train_samples=5,
            save_dir="/tmp/shakespeare_single_test"
        )
        
        print(f"\n✓ Generated single train dataset: {len(results['train'])} samples")
        
    except Exception as e:
        print(f"Error during single mode testing: {e}")
    
    print("\n" + "="*50)
    print("Shakespeare dataset testing complete!")
    print("\nNote: Full functionality requires:")
    print("1. Kaggle API credentials configured")
    print("2. Internet connection for downloading dataset")
    print("3. Sufficient disk space for the Shakespeare plays CSV")
