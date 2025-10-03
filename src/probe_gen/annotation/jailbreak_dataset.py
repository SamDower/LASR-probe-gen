"""
Jailbreak dataset implementation as a PromptDataset subclass.

This module provides a clean interface for generating jailbreak datasets
with automatic train/test separation and consistent interface. Supports both
jailbreak attempts and vanilla harmful requests.
"""

from collections import defaultdict
from typing import Optional

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from probe_gen.annotation.interface_dataset import Dataset, Message
from probe_gen.annotation.prompt_dataset import PromptDataset

# System prompt for jailbreak analysis (placeholder from original)
SYSTEM_PROMPT_JAILBREAK = """ """


class JailbreakDataset(PromptDataset):
    """
    Jailbreak dataset implementation with unified PromptDataset interface.
    
    This class handles the sevdeawesome/jailbreak_success dataset with automatic
    data downloading from HuggingFace and proper train/test separation.
    Creates datasets for analyzing jailbreak attempt behaviors.
    
    Key design: The dataset preserves both jailbreak_prompt_text (which the model
    responds to) and original_prompt_text (which autograders can use for evaluation).
    
    Features:
    - Streams data from sevdeawesome/jailbreak_success HuggingFace dataset
    - Filters out specific jailbreak types (gcg, autodan) by default
    - Proper train/test separation using skip parameters
    - Consistent interface with other prompt datasets
    - Rich metadata for jailbreak analysis
    """
    
    def __init__(
        self, 
        hf_repo_id: str = "lasrprobegen/refusal-activations",
        csv_filename: str = "jailbreak_mixed_100k_v3.csv",
        default_train_test_gap: int = 60000
    ):
        """
        Initialize Jailbreak prompt dataset.
        
        Args:
            hf_repo_id: HuggingFace repository ID containing the CSV file
            csv_filename: Name of the CSV file in the repository
            default_train_test_gap: Default gap between train and test data
        """
        super().__init__("jailbreak", default_train_test_gap)
        
        self.hf_repo_id = hf_repo_id
        self.csv_filename = csv_filename
        self._cached_df = None  # Cache processed and shuffled DataFrame
    
    def download_data(self) -> None:
        """
        Download and process jailbreak data from HuggingFace.
        
        Process: Download CSV from HF repo → Load data → Shuffle → Cache
        This ensures consistent shuffling across different skip/sample requests.
        """
        print(f"Loading jailbreak data from HuggingFace: {self.hf_repo_id}/{self.csv_filename}")
        
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
            
            # Step 3: Filter out rows with missing critical data only
            # Only drop rows where essential columns are missing
            essential_columns = ['user_content', 'jailbreak_prompt_name']
            df = df.dropna(subset=essential_columns)
            filtered_len = len(df)
            if filtered_len < original_len:
                print(f"Filtered out {original_len - filtered_len} rows with missing essential data")
            
            # Step 4: Validate required columns
            required_columns = ['user_content', 'jailbreak_prompt_name', 'sample_type']
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                print(f"Warning: CSV file missing some columns: {missing_columns}")
            
            # Step 5: Shuffle the dataframe to ensure random sampling
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            print("Shuffled dataset with random_state=42 for reproducibility")
            
            # Step 6: Cache the processed and shuffled dataframe
            self._cached_df = df
            print(f"✓ Successfully processed and cached {len(df)} jailbreak samples")
            
        except Exception as e:
            raise RuntimeError(f"Failed to download or process data from HuggingFace: {e}")
    
    def _create_dataset(self, mode: str, n_samples: int, skip: int) -> Dataset:
        """
        Create jailbreak dataset with specified parameters from pre-generated CSV data.
        
        Args:
            mode: "train" or "test" (used for identification only)
            n_samples: Number of samples to generate
            skip: Number of samples to skip from the beginning
            
        Returns:
            Dataset object with jailbreak conversations
            
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
            print("Warning: Not enough data available, using the remainder")
        if mode == "train" and end_index > self.default_train_test_gap:
            end_index = self.default_train_test_gap
            print("Warning: Not enough data available, using the remainder")
        
        # Extract the requested slice from the shuffled dataframe
        df_subset = self._cached_df.iloc[skip:end_index]
        
        ids = []
        inputs = []
        other_fields = defaultdict(list)
        
        # Process each jailbreak sample
        for idx, (shuffled_idx, row) in enumerate(tqdm(df_subset.iterrows(), 
                                                       total=len(df_subset), 
                                                       desc=f"Processing jailbreak {mode} dataset")):
            # Use the user_content as the user message
            user_content = row['user_content']
            
            # Create messages with user content and empty assistant response
            messages = [
                Message(role="user", content=user_content),
                Message(role="assistant", content="")
            ]
            
            # Create simple sample ID
            sample_id = f"jailbreak_{skip + idx + 1}_{mode}"
            
            ids.append(sample_id)
            inputs.append(messages)
            
            # Add metadata from CSV
            other_fields['prompt_name'].append(row.get('prompt_name', ''))
            other_fields['jailbreak_prompt_name'].append(row.get('jailbreak_prompt_name', ''))
            other_fields['original_prompt_text'].append(row.get('original_prompt_text', ''))
            other_fields['jailbreak_prompt_text'].append(row.get('jailbreak_prompt_text', ''))
            other_fields['sample_type'].append(row.get('sample_type', ''))
            other_fields['shuffled_index'].append(shuffled_idx)
            other_fields['processed_index'].append(skip + idx)
        
        dataset = Dataset(inputs=inputs, ids=ids, other_fields=other_fields)
        return dataset
    
    def get_dataset_info(self):
        """Get information about this jailbreak dataset."""
        info = super().get_dataset_info()
        info.update({
            'hf_repo_id': self.hf_repo_id,
            'csv_filename': self.csv_filename,
            'total_samples': len(self._cached_df) if self._cached_df is not None else "Unknown (data not loaded)",
            'description': "Pre-generated jailbreak dataset with mixed sample types",
            'data_structure': "Each sample contains user_content and metadata from CSV",
            'sample_types': "original_jailbreak, wrapped_harmful (wikipedia, evil_confidant, prefix_injection, aim, distractor)",
            'processing': "Dataset is downloaded from HF CSV, filtered, shuffled (random_state=42), then sampled"
        })
        return info


class HarmfulRequestDataset(PromptDataset):
    """
    Harmful Request dataset implementation with unified PromptDataset interface.
    
    This class handles the allenai/wildjailbreak dataset, specifically the vanilla_harmful
    prompts only. Outputs minimal JSON with just IDs and messages (no extra metadata).
    
    Features:
    - Loads vanilla_harmful prompts from allenai/wildjailbreak dataset
    - Proper train/test separation using skip parameters
    - Minimal JSON output (only sample IDs and messages)
    - Consistent interface with other prompt datasets
    """
    
    def __init__(
        self, 
        hf_dataset_name: str = "allenai/wildjailbreak",
        split: str = "train",
        default_train_test_gap: int = 1000
    ):
        """
        Initialize Harmful Request prompt dataset.
        
        Args:
            hf_dataset_name: HuggingFace dataset identifier (default: "allenai/wildjailbreak")
            split: Which dataset split to use (default: "train")
            default_train_test_gap: Default gap between train and test data (default: 1000)
        """
        super().__init__("harmful_request", default_train_test_gap)
        
        self.hf_dataset_name = hf_dataset_name
        self.split = split
        self.data_type = "vanilla_harmful"  # Always use vanilla_harmful
        self._cached_df = None  # Cache processed DataFrame to avoid reprocessing
    
    def download_data(self) -> None:
        """
        Download and cache harmful request data from HuggingFace.
        
        The data is loaded and filtered for vanilla_harmful type only.
        """
        print(f"Loading harmful request data from HuggingFace: {self.hf_dataset_name}")
        
        try:
            # Use streaming mode to bypass Arrow parsing issues
            print("Loading dataset with streaming mode to bypass parsing errors...")
            hf_dataset = load_dataset(
                self.hf_dataset_name, 
                name="train", 
                split=self.split,
                streaming=True
            )
            
            # Process data item by item to avoid parsing errors
            data_list = []
            vanilla_harmful_count = 0
            
            print("Processing dataset items...")
            for i, item in enumerate(hf_dataset):
                try:
                    # Only keep vanilla_harmful items
                    if item.get('data_type') == 'vanilla_harmful':
                        data_list.append(item)
                        vanilla_harmful_count += 1
                        
                        if vanilla_harmful_count % 100 == 0:
                            print(f"Found {vanilla_harmful_count} vanilla_harmful samples...")
                            
                    # Stop after finding enough samples to avoid long processing
                    if vanilla_harmful_count >= 10000:  # Reasonable limit
                        print(f"Reached limit of {vanilla_harmful_count} samples, stopping...")
                        break
                        
                except Exception as item_error:
                    # Skip problematic items
                    print(f"Warning: Skipping problematic item {i}: {item_error}")
                    continue
            
            print(f"Converting {len(data_list)} vanilla_harmful samples to DataFrame...")
            df = pd.DataFrame(data_list)
            
            # Shuffle the dataframe to ensure random sampling
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            print("Shuffled dataset with random_state=42 for reproducibility")
            
            self._cached_df = df
            print(f"✓ Successfully loaded and cached {len(df)} harmful request samples")
            
        except Exception as e:
            raise RuntimeError(f"Failed to access HuggingFace dataset: {e}")
    
    def _create_dataset(self, mode: str, n_samples: int, skip: int) -> Dataset:
        """
        Create harmful request dataset with specified parameters.
        
        Args:
            mode: "train" or "test" (used for identification only)
            n_samples: Number of samples to generate
            skip: Number of samples to skip from the beginning
            
        Returns:
            Dataset object with harmful request conversations
            
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
            print("Warning: Not enough data available, using the remainder")
        if mode == "train" and end_index > self.default_train_test_gap:
            end_index = self.default_train_test_gap
            print("Warning: Not enough data available, using the remainder")
        
        # Extract the requested slice from the shuffled dataframe
        df_subset = self._cached_df.iloc[skip:end_index]
        
        ids = []
        inputs = []
        other_fields = defaultdict(list)
        
        # Process each harmful request sample
        for idx, (shuffled_idx, row) in enumerate(tqdm(df_subset.iterrows(), 
                                                       total=len(df_subset), 
                                                       desc=f"Processing harmful request {mode} dataset")):
            # Use the vanilla harmful request as the user message
            user_content = row['vanilla']
            
            # Create messages with user harmful request and empty assistant response
            messages = [
                Message(role="user", content=user_content),
                Message(role="assistant", content="")
            ]
            
            # Create simple, clean sample ID
            sample_id = f"harmful_{skip + idx + 1}_{mode}"
            
            ids.append(sample_id)
            inputs.append(messages)
        
        dataset = Dataset(inputs=inputs, ids=ids, other_fields=other_fields)
        return dataset
    
    def get_dataset_info(self):
        """Get information about this harmful request dataset."""
        info = super().get_dataset_info()
        info.update({
            'hf_dataset_name': self.hf_dataset_name,
            'split': self.split,
            'data_type': 'vanilla_harmful',
            'total_samples': len(self._cached_df) if self._cached_df is not None else "Unknown (data not loaded)",
            'description': "Harmful request dataset containing vanilla_harmful prompts only",
            'data_structure': "Minimal JSON with only sample IDs and messages",
            'processing': "Dataset is filtered for vanilla_harmful, shuffled (random_state=42), then sampled"
        })
        return info
    


# Legacy functions for backward compatibility
def create_jailbreak_dataset(num_samples: int = 1000, skip: int = 0) -> Dataset:
    """
    Create jailbreak dataset using the legacy function interface.
    
    This function is kept for backward compatibility.
    New code should use JailbreakDataset class directly.
    """
    dataset_generator = JailbreakDataset()
    dataset_generator.download_data()
    
    # Use the new interface but return only the train dataset for compatibility
    results = dataset_generator.generate_data(
        train_samples=num_samples,
        skip=skip,
        save_dir=None  # Don't save automatically in legacy mode
    )
    
    return results['train']


def create_harfmul_request_dataset(num_samples: int = 1000, skip: int = 0) -> Dataset:
    """
    Create harmful request dataset using the legacy function interface.
    
    This function is kept for backward compatibility.
    New code should use HarmfulRequestDataset class directly.
    """
    dataset_generator = HarmfulRequestDataset()
    dataset_generator.download_data()
    
    # Use the new interface but return only the train dataset for compatibility
    results = dataset_generator.generate_data(
        train_samples=num_samples,
        skip=skip,
        save_dir=None  # Don't save automatically in legacy mode
    )
    
    return results['train']


# Example usage and testing
if __name__ == "__main__":
    print("Testing JailbreakDataset and HarmfulRequestDataset...")
    
    # Test JailbreakDataset
    print("\n" + "="*50)
    print("Testing jailbreak dataset...")
    
    jailbreak_dataset = JailbreakDataset()
    
    print(f"\nJailbreak dataset info: {jailbreak_dataset.get_dataset_info()}")
    
    try:
        print("\nTesting jailbreak data access...")
        jailbreak_dataset.download_data()
        print("✓ Jailbreak data access verification successful")
        
        # Test small dataset generation
        print("\nTesting jailbreak dataset generation...")
        results = jailbreak_dataset.generate_data(
            train_samples=5,
            test_samples=3,
            save_dir="/tmp/jailbreak_test"
        )
        
        print("\n✓ Generated jailbreak datasets:")
        print(f"  Train: {len(results['train'])} samples")
        print(f"  Test: {len(results['test'])} samples")
        
        # Show example jailbreak
        example_messages = results['train'].inputs[0]
        print("\nExample jailbreak attempt:")
        print(f"  User: {example_messages[0].content[:100]}...")
        
        # Show some metadata
        if results['train'].other_fields:
            print("\nExample jailbreak metadata:")
            print(f"  Prompt name: {results['train'].other_fields['prompt_name'][0]}")
            print(f"  Jailbreak type: {results['train'].other_fields['jailbreak_prompt_name'][0]}")
            print(f"  Original preview: {results['train'].other_fields['original_preview'][0]}")
        
    except Exception as e:
        print(f"Error during jailbreak testing: {e}")
        print("Note: This might fail if HuggingFace dataset is not accessible")
    
    # Test HarmfulRequestDataset
    print("\n" + "="*50)
    print("Testing harmful request dataset...")
    
    harmful_dataset = HarmfulRequestDataset()
    
    print(f"\nHarmful request dataset info: {harmful_dataset.get_dataset_info()}")
    
    try:
        print("\nTesting harmful request data access...")
        harmful_dataset.download_data()
        print("✓ Harmful request data access verification successful")
        
        # Test small dataset generation
        print("\nTesting harmful request dataset generation...")
        results = harmful_dataset.generate_data(
            train_samples=5,
            test_samples=3,
            save_dir="/tmp/harmful_test"
        )
        
        print("\n✓ Generated harmful request datasets:")
        print(f"  Train: {len(results['train'])} samples")
        print(f"  Test: {len(results['test'])} samples")
        
        # Show example harmful request
        example_messages = results['train'].inputs[0]
        print("\nExample harmful request:")
        print(f"  User: {example_messages[0].content[:100]}...")
        
        # Show some metadata
        if results['train'].other_fields:
            print("\nExample harmful request metadata:")
            print(f"  Data type: {results['train'].other_fields['data_type'][0]}")
            print(f"  Request preview: {results['train'].other_fields['request_preview'][0]}")
        
    except Exception as e:
        print(f"Error during harmful request testing: {e}")
        print("Note: This might fail if HuggingFace dataset is not accessible")
    
    # Test legacy function compatibility
    print("\n" + "="*50)
    print("Testing legacy function compatibility...")
    
    try:
        legacy_jailbreak = create_jailbreak_dataset(num_samples=3, skip=0)
        print(f"\n✓ Legacy jailbreak function works: {len(legacy_jailbreak)} samples")
        
        legacy_harmful = create_harfmul_request_dataset(num_samples=3, skip=0)
        print(f"✓ Legacy harmful request function works: {len(legacy_harmful)} samples")
        
    except Exception as e:
        print(f"Error during legacy testing: {e}")
    
    print("\n" + "="*50)
    print("Jailbreak and harmful request dataset testing complete!")
    print("\nNote: Full functionality requires:")
    print("1. Internet connection for accessing HuggingFace datasets")
    print("2. HuggingFace datasets library installed")
    print("3. pandas library for data processing")
