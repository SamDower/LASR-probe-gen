"""
Writing Prompts dataset implementation as a PromptDataset subclass.

This module provides a clean interface for generating creative writing datasets
with automatic train/test separation and consistent interface. Uses the HuggingFace
euclaise/writingprompts dataset for creating creative writing prompts.
"""

import pandas as pd

from probe_gen.annotation.interface_dataset import Dataset, Message
from probe_gen.annotation.prompt_dataset import PromptDataset


class WritingPromptsDataset(PromptDataset):
    """
    Writing Prompts dataset implementation with unified PromptDataset interface.
    
    This class handles the euclaise/writingprompts dataset from HuggingFace with automatic
    data downloading and proper train/test separation. Creates creative writing prompts
    based on writing prompts and their corresponding stories.
    
    Features:
    - Automatic download from HuggingFace (euclaise/writingprompts)
    - Uses raw prompts directly without additional instructions
    - Proper train/test separation using skip parameters
    - Consistent interface with other prompt datasets
    """
    
    def __init__(
        self, 
        hf_dataset_name: str = "euclaise/writingprompts",
        split: str = "train",
        default_train_test_gap: int = 10000
    ):
        """
        Initialize Writing Prompts dataset.
        
        Args:
            hf_dataset_name: HuggingFace dataset identifier
            split: Dataset split to use ("train", "test", "validation")
            default_train_test_gap: Default gap between train and test data
        """
        super().__init__("writingprompts", default_train_test_gap)
        
        self.hf_dataset_name = hf_dataset_name
        self.split = split
        self._cached_df = None  # Cache processed data to avoid reprocessing
        self._data_loaded = False  # Track if data has been successfully loaded
        
        # No instruction templates - use raw prompts directly
        self.instruction_templates = {}
    
    def download_data(self) -> None:
        """
        Download Writing Prompts dataset from HuggingFace if not already available.
        
        Loads the dataset, filters by prompt length, and caches the processed data.
        """
        print(f"Loading Writing Prompts dataset from HuggingFace: {self.hf_dataset_name}")
        
        try:
            # Import datasets here to avoid conflict with local datasets.py
            from datasets import load_dataset
            
            # Load the HuggingFace dataset
            hf_dataset = load_dataset(self.hf_dataset_name, split=self.split)
            df = pd.DataFrame(hf_dataset)
            original_len = len(df)
            print(f"Loaded {original_len} total samples from HuggingFace")
            
            # Use all prompts without filtering by length
            print(f"Using all {len(df)} samples from the dataset")
            
            # Shuffle the dataframe to ensure random sampling
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            print("Shuffled dataset with random_state=42 for reproducibility")
            
            # Cache the processed dataframe
            self._cached_df = df
            self._data_loaded = True
            print(f"✓ Successfully processed and cached {len(df)} writing prompt samples")
            
        except Exception as e:
            print(f"Warning: Failed to load Writing Prompts dataset: {e}")
            # Create a fallback empty dataset to prevent crashes
            self._cached_df = pd.DataFrame(columns=['prompt', 'story'])
            self._data_loaded = False
            print("Created empty fallback dataset")
    
    def _create_dataset(self, mode: str, n_samples: int, skip: int) -> Dataset:
        """
        Create Writing Prompts dataset with specified parameters.
        
        Args:
            mode: "train" or "test" (used for identification only)
            n_samples: Number of samples to generate
            skip: Number of samples to skip from the beginning
            
        Returns:
            Dataset object with creative writing prompts
            
        Raises:
            ValueError: If not enough samples are available
        """
        # Ensure data is processed and cached
        if self._cached_df is None:
            raise ValueError("Data not downloaded. Call download_data() first.")
        
        # Check if data loading was successful
        if not self._data_loaded:
            print("Warning: Writing Prompts dataset failed to load from HuggingFace")
            # Return empty dataset
            return Dataset(inputs=[], ids=[], other_fields={})
        
        # Check if we have any data at all
        if len(self._cached_df) == 0:
            print("Warning: No data available in Writing Prompts dataset")
            # Return empty dataset
            return Dataset(inputs=[], ids=[], other_fields={})
        
        # Check if we have enough samples after skipping
        total_available = len(self._cached_df)
        end_index = skip + n_samples
        if end_index > total_available:
            end_index = total_available
            print(f"Warning: Not enough data available, using {end_index - skip} samples instead of {n_samples}")
        
        # Extract the requested slice
        selected_data = self._cached_df.iloc[skip:end_index]
        
        ids = []
        inputs = []
        other_fields = {
            "original_index": [],
            "processed_index": [],
            "prompt_length": [],
            "story_length": []
        }
        
        # Process the selected samples
        for processed_ix, (_, row) in enumerate(selected_data.iterrows()):
            prompt_text = row['prompt']
            story_text = row['story']
            
            # Use raw prompt directly without any instruction template
            full_prompt = prompt_text
            
            # Create conversation messages with user prompt and empty assistant response
            messages = [
                Message(role="user", content=full_prompt),
                Message(role="assistant", content="")
            ]
            
            # Create unique sample ID
            sample_id = f"writingprompts_{skip + processed_ix}_{mode}"
            
            ids.append(sample_id)
            inputs.append(messages)
            
            # Add metadata
            other_fields["original_index"].append(skip + processed_ix)
            other_fields["processed_index"].append(processed_ix)
            other_fields["prompt_length"].append(len(prompt_text))
            other_fields["story_length"].append(len(story_text))
        
        dataset = Dataset(inputs=inputs, ids=ids, other_fields=other_fields)
        return dataset
    
    def get_dataset_info(self):
        """Get information about this Writing Prompts dataset."""
        info = super().get_dataset_info()
        info.update({
            'hf_dataset_name': self.hf_dataset_name,
            'split': self.split,
            'total_processed_samples': len(self._cached_df) if self._cached_df is not None else "Unknown (data not processed)",
            'data_loaded_successfully': self._data_loaded,
            'description': "Creative writing prompts dataset based on euclaise/writingprompts"
        })
        return info


# Example usage and testing
if __name__ == "__main__":
    print("Testing WritingPromptsDataset...")
    
    # Test dataset initialization
    print("\n" + "="*50)
    print("Testing Writing Prompts dataset...")
    
    writingprompts_dataset = WritingPromptsDataset(
        min_prompt_length=15,  # Lower threshold for testing
        instruction_type="creative_writing"
    )
    
    print(f"\nDataset info: {writingprompts_dataset.get_dataset_info()}")
    
    # Test data download
    try:
        print("\nTesting data download...")
        writingprompts_dataset.download_data()
        print("✓ Data download successful")
        
        # Test small dataset generation
        print("\nTesting dataset generation...")
        results = writingprompts_dataset.generate_data(
            mode="train",
            n_samples=10,
            output_file="/tmp/writingprompts_test_train.json"
        )
        
        print(f"\n✓ Generated train dataset: {results} samples")
        
        # Show example prompt
        if results:
            print("\nExample usage:")
            print("To generate a dataset, use:")
            print("writingprompts_dataset.generate_data(")
            print("    mode='train',")
            print("    n_samples=1000,")
            print("    output_file='path/to/output.json'")
            print(")")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Note: This might fail if HuggingFace datasets is not installed")
    
    print("\n" + "="*50)
    print("Writing Prompts dataset testing complete!")
    print("\nNote: Full functionality requires:")
    print("1. HuggingFace datasets library installed")
    print("2. Internet connection for downloading dataset")
    print("3. Sufficient disk space for the Writing Prompts dataset")
