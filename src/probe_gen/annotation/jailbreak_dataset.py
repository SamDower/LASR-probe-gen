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
        hf_dataset_name: str = "sevdeawesome/jailbreak_success",
        split: str = "train",
        excluded_jailbreak_types: Optional[list] = None,
        default_train_test_gap: int = 1000
    ):
        """
        Initialize Jailbreak prompt dataset.
        
        Args:
            hf_dataset_name: HuggingFace dataset identifier
            split: Which dataset split to use ("train")
            excluded_jailbreak_types: List of jailbreak types to exclude (default: ['gcg', 'autodan', 'original'])
            default_train_test_gap: Default gap between train and test data
        """
        super().__init__("jailbreak", default_train_test_gap)
        
        self.hf_dataset_name = hf_dataset_name
        self.split = split
        self.excluded_jailbreak_types = excluded_jailbreak_types or ['gcg', 'autodan', 'original']
        self._cached_df = None  # Cache processed and shuffled DataFrame
    
    def download_data(self) -> None:
        """
        Download and process jailbreak data from HuggingFace.
        
        Process: Load entire dataset -> Filter excluded types -> Shuffle -> Cache
        This ensures consistent shuffling across different skip/sample requests.
        """
        print(f"Loading jailbreak data from HuggingFace: {self.hf_dataset_name}")
        
        try:
            # Step 1: Load the entire HuggingFace dataset
            hf_dataset = load_dataset(self.hf_dataset_name, split=self.split)
            df = pd.DataFrame(hf_dataset)
            original_len = len(df)
            print(f"Loaded {original_len} total samples from HuggingFace")
            
            # Step 2: Filter out excluded jailbreak types
            if self.excluded_jailbreak_types:
                df = df[~df['jailbreak_prompt_name'].isin(self.excluded_jailbreak_types)]
                filtered_len = len(df)
                print(f"Filtered out {original_len - filtered_len} samples with excluded jailbreak types: {self.excluded_jailbreak_types}")
                print(f"Remaining samples after filtering: {filtered_len}")
            
            # Step 3: Shuffle the dataframe to ensure random sampling
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            print("Shuffled dataset with random_state=42 for reproducibility")
            
            # Step 4: Cache the processed and shuffled dataframe
            self._cached_df = df
            print(f"✓ Successfully processed and cached {len(df)} jailbreak samples")
            
        except Exception as e:
            raise RuntimeError(f"Failed to access HuggingFace dataset: {e}")
    
    def _create_dataset(self, mode: str, n_samples: int, skip: int) -> Dataset:
        """
        Create jailbreak dataset with specified parameters.
        
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

        # Check if we have enough arguments after skipping
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
            # Use the jailbreak_prompt_text as the user message (contains the jailbreak attempt)
            user_content = row['jailbreak_prompt_text']
            
            # Create messages with user jailbreak attempt and empty assistant response
            messages = [
                Message(role="user", content=user_content),
                Message(role="assistant", content="")
            ]
            
            # Create unique sample ID using shuffled index to maintain uniqueness
            prompt_name = row['prompt_name']
            jailbreak_type = row['jailbreak_prompt_name']
            sample_id = f"jailbreak_{shuffled_idx}_{prompt_name}_{jailbreak_type}_{mode}"
            
            ids.append(sample_id)
            inputs.append(messages)
            
            # Add rich metadata for jailbreak analysis
            other_fields['prompt_name'].append(prompt_name)
            other_fields['jailbreak_prompt_name'].append(jailbreak_type)
            other_fields['original_prompt_text'].append(row['original_prompt_text'])
            other_fields['jailbreak_prompt_text'].append(row['jailbreak_prompt_text'])
            other_fields['shuffled_index'].append(shuffled_idx)  # Track position in shuffled dataset
            other_fields['processed_index'].append(skip + idx)   # Track position in processed subset
        
        dataset = Dataset(inputs=inputs, ids=ids, other_fields=other_fields)
        return dataset
    
    def get_dataset_info(self):
        """Get information about this jailbreak dataset."""
        info = super().get_dataset_info()
        info.update({
            'hf_dataset_name': self.hf_dataset_name,
            'split': self.split,
            'excluded_jailbreak_types': self.excluded_jailbreak_types,
            'total_samples': len(self._cached_df) if self._cached_df is not None else "Unknown (data not loaded)",
            'description': "Jailbreak attempts dataset for analyzing adversarial prompt behaviors",
            'data_structure': "Each sample contains both original prompt and jailbreak version",
            'processing': "Dataset is filtered, shuffled (random_state=42), then sampled"
        })
        return info


class HarmfulRequestDataset(PromptDataset):
    """
    Harmful Request dataset implementation with unified PromptDataset interface.
    
    This class handles the allenai/wildjailbreak dataset, specifically the vanilla_harmful
    prompts, with automatic data downloading from HuggingFace and proper train/test separation.
    
    Features:
    - Loads vanilla_harmful prompts from allenai/wildjailbreak dataset
    - Proper train/test separation using skip parameters
    - Consistent interface with other prompt datasets
    - Simpler structure than jailbreak attempts (direct harmful requests)
    """
    
    def __init__(
        self, 
        hf_dataset_name: str = "allenai/wildjailbreak",
        split: str = "train",
        data_type: str = "vanilla_harmful",
        default_train_test_gap: int = 1000
    ):
        """
        Initialize Harmful Request prompt dataset.
        
        Args:
            hf_dataset_name: HuggingFace dataset identifier
            split: Which dataset split to use ("train")
            data_type: Type of data to filter for (default: "vanilla_harmful")
            default_train_test_gap: Default gap between train and test data
        """
        super().__init__("harmful_request", default_train_test_gap)
        
        self.hf_dataset_name = hf_dataset_name
        self.split = split
        self.data_type = data_type
        self._cached_df = None  # Cache processed DataFrame to avoid reprocessing
    
    def download_data(self) -> None:
        """
        Download and cache harmful request data from HuggingFace.
        
        The data is loaded and filtered for the specified data_type.
        """
        print(f"Loading harmful request data from HuggingFace: {self.hf_dataset_name}")
        
        try:
            # Load the WildJailbreak dataset
            hf_dataset = load_dataset(self.hf_dataset_name, self.split, delimiter="\t", keep_default_na=False)
            
            # Convert to DataFrame and filter for specified data type
            df = pd.DataFrame(hf_dataset['train'])
            original_len = len(df)
            df = df[df['data_type'] == self.data_type]
            filtered_len = len(df)
            
            print(f"Filtered {original_len} samples to {filtered_len} samples of type '{self.data_type}'")
            
            self._cached_df = df
            print(f"✓ Successfully loaded {len(df)} harmful request samples")
            
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
        
        total_samples = len(self._cached_df)
        
        # Check if we have enough samples after skipping
        if skip >= total_samples:
            raise ValueError(
                f"Skip value ({skip}) is greater than or equal to total available samples ({total_samples})"
            )
        
        end_index = skip + n_samples
        if end_index > total_samples:
            available_after_skip = total_samples - skip
            raise ValueError(
                f"Not enough samples available. Requested {n_samples} samples after skipping {skip}, "
                f"but only {available_after_skip} samples available. Total samples: {total_samples}"
            )
        
        # Extract the requested slice
        df_subset = self._cached_df.iloc[skip:end_index]
        
        ids = []
        inputs = []
        other_fields = defaultdict(list)
        
        # Process each harmful request sample
        for idx, (original_idx, row) in enumerate(tqdm(df_subset.iterrows(), 
                                                       total=len(df_subset), 
                                                       desc=f"Processing harmful request {mode} dataset")):
            # Use the vanilla harmful request as the user message
            user_content = row['vanilla']
            
            # Create messages with user harmful request and empty assistant response
            messages = [
                Message(role="user", content=user_content),
                Message(role="assistant", content="")
            ]
            
            # Create unique sample ID
            row_id = row.get('id', f'noid_{original_idx}')
            sample_id = f"harmful_{original_idx}_{row_id}_{mode}"
            
            ids.append(sample_id)
            inputs.append(messages)
            
            # Add metadata
            other_fields['data_type'].append(row['data_type'])
            other_fields['vanilla_request'].append(row['vanilla'])
            other_fields['original_index'].append(original_idx)
            other_fields['processed_index'].append(skip + idx)
            other_fields['row_id'].append(row_id)
            
            # Add preview for easier inspection
            other_fields['request_preview'].append(
                row['vanilla'][:100] + "..." if len(row['vanilla']) > 100 else row['vanilla']
            )
        
        dataset = Dataset(inputs=inputs, ids=ids, other_fields=other_fields)
        return dataset
    
    def get_dataset_info(self):
        """Get information about this harmful request dataset."""
        info = super().get_dataset_info()
        info.update({
            'hf_dataset_name': self.hf_dataset_name,
            'split': self.split,
            'data_type': self.data_type,
            'total_samples': len(self._cached_df) if self._cached_df is not None else "Unknown (data not loaded)",
            'description': f"Harmful request dataset containing {self.data_type} prompts",
            'data_structure': "Direct harmful requests without jailbreak attempts"
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
