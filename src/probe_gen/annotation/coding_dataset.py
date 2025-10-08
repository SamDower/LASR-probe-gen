"""
Coding dataset implementation as a PromptDataset subclass.

This module provides a clean interface for generating coding datasets
with automatic train/test separation and consistent interface. Supports
coding questions from HuggingFace datasets with proper OpenAI format.
"""

from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from probe_gen.annotation.interface_dataset import Dataset, Message
from probe_gen.annotation.prompt_dataset import PromptDataset


class CodingQuestionsDataset(PromptDataset):
    """
    Coding questions dataset implementation 
    
    Features:
    - Uses combined dataset from Muennighoff/mbpp and openai/openai_humaneval HuggingFace datasets
    - Processes CSV data with 'text' and 'source' columns
    - Creates proper OpenAI format messages with user/assistant roles
    """
    
    def __init__(
        self, 
        hf_repo_id: str = "lasrprobegen/lists-activations",
        csv_filename: str = "coding_questions.csv",
        default_train_test_gap: int = 0
    ):
        """
        Initialize Coding questions dataset.
        
        Args:
            hf_repo_id: HuggingFace repository ID containing the CSV file
            csv_filename: Name of the CSV file in the repository
            default_train_test_gap: Default gap between train and test data (default is 0 because this is meant to be used as a test set)
        """
        super().__init__("coding", default_train_test_gap)
        
        self.hf_repo_id = hf_repo_id
        self.csv_filename = csv_filename
        self._cached_df = None  # Cache processed and shuffled DataFrame
    
    def download_data(self) -> None:
        """
        Download and process coding questions data from HuggingFace.
        
        Process: Download CSV from HF repo → Load data → Shuffle → Cache
        This ensures consistent shuffling across different skip/sample requests.
        """
        print(f"Loading coding questions data from HuggingFace: {self.hf_repo_id}/{self.csv_filename}")
        print(self.hf_repo_id)
        print(self.csv_filename)
        
        try:
            # Step 1: Download CSV from HuggingFace repository
            from huggingface_hub import hf_hub_download
            
            csv_path = hf_hub_download(
                repo_id=self.hf_repo_id,
                filename=self.csv_filename,
                repo_type="dataset"
            )
            print(f"Downloaded CSV file from HuggingFace to: {csv_path}")
            
            # Step 2: Load CSV data
            df = pd.read_csv(csv_path, index_col=None)
            original_len = len(df)
            print(f"Loaded {original_len} total samples from CSV")
            
            # Step 3: Validate required columns
            required_columns = ['text', 'source']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {list(df.columns)}")
            
            # Step 4: Shuffle the dataframe to ensure random sampling
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            print("Shuffled dataset with random_state=42 for reproducibility")
            
            # Step 5: Cache the processed and shuffled dataframe
            self._cached_df = df
            print(f"✓ Successfully processed and cached {len(df)} coding samples")
            
        except Exception as e:
            raise RuntimeError(f"Failed to download or process data from HuggingFace: {e}")
    
    def _create_dataset(self, mode: str, n_samples: int, skip: int) -> Dataset:
        """
        Create coding questions dataset with specified parameters from pre-generated CSV data.
        
        Args:
            mode: "train" or "test" (used for identification only)
            n_samples: Number of samples to generate
            skip: Number of samples to skip from the beginning
            
        Returns:
            Dataset object with coding questions conversations in OpenAI format
            
        Raises:
            ValueError: If not enough samples are available
        """
        if self._cached_df is None:
            raise ValueError("Data not loaded. Call download_data() first.")
        
        # Check if we have enough samples after skipping
        total_available = len(self._cached_df)
        end_index = skip + n_samples
        if end_index > total_available:
            end_index = total_available
            print(f"Warning: Not enough data available, using {end_index - skip} samples instead of {n_samples}")
        if mode == "train" and end_index > self.default_train_test_gap:
            end_index = self.default_train_test_gap
            print(f"Warning: Train data limited by train_test_gap, using {end_index - skip} samples")
        
        # Extract the requested slice from the shuffled dataframe
        df_subset = self._cached_df.iloc[skip:end_index]
        
        ids = []
        inputs = []
        other_fields = defaultdict(list)
        
        # Process each coding questions sample
        for idx, (shuffled_idx, row) in enumerate(tqdm(df_subset.iterrows(), 
                                                       total=len(df_subset), 
                                                       desc=f"Processing coding questions {mode} dataset")):
            # Use the text column as the user message (coding question)
            user_content = row['text']
            
            # Create messages in OpenAI format with user question and empty assistant response
            messages = [
                Message(role="user", content=user_content),
                Message(role="assistant", content="")
            ]
            
            # Create simple sample ID
            sample_id = f"coding_{skip + idx + 1}_{mode}"
            
            ids.append(sample_id)
            inputs.append(messages)
            
            # Add metadata from CSV - preserve original text and source
            other_fields['text'].append(row['text'])
            other_fields['source'].append(row['source'])
            
            # Add any additional columns from the CSV as metadata
            for col in df_subset.columns:
                if col not in ['text', 'source']:  # Skip already processed columns
                    other_fields[col].append(row.get(col, ''))
        
        dataset = Dataset(inputs=inputs, ids=ids, other_fields=other_fields)
        return dataset
    
    def get_dataset_info(self):
        """Get information about this coding questions dataset."""
        info = super().get_dataset_info()
        info.update({
            'hf_repo_id': self.hf_repo_id,
            'csv_filename': self.csv_filename,
            'total_samples': len(self._cached_df) if self._cached_df is not None else "Unknown (data not loaded)",
            'description': "Coding questions dataset from HuggingFace with text and source columns",
            'data_structure': "Each sample contains coding question text and source (humaneval/mbpp)",
            'required_columns': "text (coding question), source (dataset origin)",
            'message_format': "OpenAI format with user question and empty assistant response",
            'processing': "Dataset is downloaded from HF CSV, shuffled (random_state=42), then sampled"
        })
        return info


# Legacy function for backward compatibility
def create_coding_dataset(num_samples: int = 1000, skip: int = 0) -> Dataset:
    """
    Create coding dataset using the legacy function interface.
    
    This function is kept for backward compatibility.
    New code should use CodingQuestionsDataset class directly.
    """
    dataset_generator = CodingQuestionsDataset()
    dataset_generator.download_data()
    
    # Use the new interface but return only the train dataset for compatibility
    dataset_generator.generate_data(
        mode="train",
        n_samples=num_samples,
        skip=skip,
        output_file="/tmp/coding_legacy.json"  # Temporary file for legacy mode
    )
    
    # Load and return the dataset
    return Dataset.from_jsonl("/tmp/coding_legacy.json")


# Example usage and testing
if __name__ == "__main__":
    print("Testing CodingQuestionsDataset...")
    
    # Test CodingQuestionsDataset
    print("\n" + "="*50)
    print("Testing coding questions dataset...")
    
    coding_dataset = CodingQuestionsDataset()
    
    print(f"\nCoding dataset info: {coding_dataset.get_dataset_info()}")
    
    try:
        print("\nTesting coding data access...")
        coding_dataset.download_data()
        print("✓ Coding data access verification successful")
        
        # Test small dataset generation
        print("\nTesting coding dataset generation...")
        success = coding_dataset.generate_data(
            mode="train",
            n_samples=5,
            output_file="/tmp/coding_test.json"
        )
        
        if success:
            # Load the generated dataset to show examples
            test_dataset = Dataset.from_jsonl("/tmp/coding_test.json")
            
            print(f"\n✓ Generated coding dataset: {len(test_dataset)} samples")
            
            # Show example coding question
            example_messages = test_dataset.inputs[0]
            print("\nExample coding question:")
            print(f"  User: {example_messages[0].content[:100]}...")
            
            # Show some metadata
            if test_dataset.other_fields:
                print("\nExample coding metadata:")
                print(f"  Source: {test_dataset.other_fields['source'][0]}")
                print(f"  Text preview: {test_dataset.other_fields['text'][0][:100]}...")
        else:
            print("Failed to generate enough samples")
        
    except Exception as e:
        print(f"Error during coding testing: {e}")
        print("Note: This might fail if HuggingFace dataset is not accessible")
    
    # Test legacy function compatibility
    print("\n" + "="*50)
    print("Testing legacy function compatibility...")
    
    try:
        legacy_coding = create_coding_dataset(num_samples=3, skip=0)
        print(f"\n✓ Legacy coding function works: {len(legacy_coding)} samples")
        
    except Exception as e:
        print(f"Error during legacy testing: {e}")
    
    print("\n" + "="*50)
    print("Coding dataset testing complete!")
    print("\nNote: Full functionality requires:")
    print("1. Internet connection for accessing HuggingFace datasets")
    print("2. HuggingFace datasets library installed")
    print("3. pandas library for data processing")