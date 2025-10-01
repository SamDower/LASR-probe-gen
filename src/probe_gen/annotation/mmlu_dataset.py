"""
MMLU dataset implementation as a PromptDataset subclass.

This module provides a clean interface for generating MMLU (Massive Multitask Language Understanding)
datasets with automatic train/test separation and consistent interface.
"""

import random
from typing import List, Optional

import pandas as pd
from datasets import load_dataset

from probe_gen.annotation.interface_dataset import Dataset, Message
from probe_gen.annotation.prompt_dataset import PromptDataset

# MMLU subcategories covering different domains
MMLU_SUBCATEGORIES = [
    "college_biology",           # Life Sciences
    "college_computer_science",  # Computer Science
    "high_school_mathematics",   # Mathematics
    "anatomy",                   # Medical/Health Sciences
    "human_sexuality",           # Human Sexuality
    "high_school_psychology",    # Psychology
    "business_ethics",           # Social Sciences/Ethics
    "logical_fallacies",         # Logical Fallacies
    "high_school_world_history", # History
    "high_school_us_history",    # History
    "management",                # Business
    "marketing",                 # Business
    "sociology",                 # Social Sciences
    "world_religions",           # Religious Studies
    "abstract_algebra",        # Abstract Reasoning
    "astronomy",               # Astronomy
    "clinical_knowledge",      # Clinical Knowledge
    "college_chemistry", 
    "college_mathematics",
    "college_physics",
    "college_medicine",
    "elementary_mathematics",
    "global_facts",
    "jurisprudence",
    "international_law",
    "miscellaneous",
    "nutrition",
    "philosophy",

    ]


class MMLUDataset(PromptDataset):
    """
    MMLU dataset implementation with unified PromptDataset interface.
    
    This class handles MMLU (Massive Multitask Language Understanding) datasets
    with automatic data downloading from HuggingFace and proper train/test separation.
    
    Features:
    - Supports all MMLU subcategories or combined datasets
    - Automatic question formatting with multiple choice options
    - Proper train/test separation using skip parameters
    - Consistent interface with other prompt datasets
    """
    
    def __init__(
        self, 
        subcategories: Optional[List[str]] = None,
        split: str = "test",
        default_train_test_gap: int = 5000
    ):
        """
        Initialize MMLU prompt dataset.
        
        Args:
            subcategories: List of MMLU subcategories to use. If None, uses all categories.
            split: Which HuggingFace split to use ("test", "dev", "val")
            default_train_test_gap: Default gap between train and test data
        """
        super().__init__("mmlu", default_train_test_gap)
        
        self.subcategories = subcategories or MMLU_SUBCATEGORIES
        self.split = split
        self._cached_data = {}  # Cache loaded datasets to avoid re-downloading
        self._combined_dataset = None  # Cache combined shuffled dataset
        
        # Validate subcategories
        invalid_categories = set(self.subcategories) - set(MMLU_SUBCATEGORIES)
        if invalid_categories:
            raise ValueError(f"Invalid MMLU subcategories: {invalid_categories}")
    
    def download_data(self) -> None:
        """
        Download MMLU data from HuggingFace for all specified subcategories.
        
        The data is cached in memory to avoid repeated downloads during the same session.
        """
        print(f"Loading MMLU data for {len(self.subcategories)} subcategories...")
        
        for subcategory in self.subcategories:
            if subcategory not in self._cached_data:
                print(f"  Loading {subcategory}...")
                try:
                    hf_dataset = load_dataset("cais/mmlu", subcategory, split=self.split)
                    self._cached_data[subcategory] = pd.DataFrame(hf_dataset)
                except Exception as e:
                    print(f"  Warning: Failed to load {subcategory}: {e}")
                    continue
        
        print(f"✓ Successfully loaded {len(self._cached_data)} MMLU subcategories")
    
    def _create_combined_shuffled_dataset(self) -> List[tuple]:
        """
        Create a combined and shuffled dataset from all subcategories.
        Uses caching to avoid regenerating the combined dataset.
        
        Returns:
            List of tuples: (formatted_question, sample_id, subcategory)
        """
        # Return cached version if available
        if self._combined_dataset is not None:
            return self._combined_dataset
        
        if not self._cached_data:
            raise ValueError("No data available. Data download may have failed.")
        
        all_samples = []
        
        # Combine all subcategories
        for subcategory in self.subcategories:
            if subcategory not in self._cached_data:
                print(f"Warning: Skipping {subcategory} - not loaded")
                continue
            
            df = self._cached_data[subcategory]
            
            # Process each sample in this subcategory
            for idx, row in df.iterrows():
                # Format the question with multiple choice options
                question = row['question']
                choices = row['choices']
                
                # Create the formatted question with choices (A, B, C, D)
                formatted_question = f"{question}\n\n"
                for j, choice_text in enumerate(choices):
                    choice_label = chr(65 + j)  # A, B, C, D
                    formatted_question += f"{choice_label}. {choice_text}\n"
                
                # Create sample tuple
                sample_id = f"mmlu_{subcategory}_{idx}"
                all_samples.append((formatted_question.strip(), sample_id, subcategory))
        
        # Shuffle the combined dataset
        random.shuffle(all_samples)
        
        # Cache the result
        self._combined_dataset = all_samples
        
        print(f"✓ Created combined shuffled dataset with {len(all_samples)} samples")
        return all_samples
    
    def _create_dataset(self, mode: str, n_samples: int, skip: int) -> Dataset:
        """
        Create MMLU dataset with specified parameters from the combined shuffled dataset.
        
        Args:
            mode: "train" or "test" (used for identification only)
            n_samples: Number of samples to generate
            skip: Number of samples to skip from the beginning
            
        Returns:
            Dataset object with MMLU questions formatted as conversations
            
        Raises:
            ValueError: If not enough samples are available
        """
        # Get the combined shuffled dataset
        combined_samples = self._create_combined_shuffled_dataset()
        
        # Check if we have enough arguments after skipping
        total_available = len(combined_samples)
        end_index = skip + n_samples
        if end_index > total_available:
            end_index = total_available
            print("Warning: Not enough data available, using the remainder")
        if mode == "train" and end_index > self.default_train_test_gap:
            end_index = self.default_train_test_gap
            print("Warning: Not enough data available, using the remainder")
        
        # Extract the requested slice from the shuffled dataset
        selected_samples = combined_samples[skip:end_index]
        
        all_ids = []
        all_inputs = []
        
        # Process the selected samples
        for formatted_question, sample_id, subcategory in selected_samples:
            # Create conversation messages with user question and empty assistant response
            messages = [
                Message(role="user", content=formatted_question),
                Message(role="assistant", content="")
            ]
            
            # Update sample ID to include the mode
            updated_sample_id = f"{sample_id}_{mode}"
            
            all_ids.append(updated_sample_id)
            all_inputs.append(messages)
        
        dataset = Dataset(inputs=all_inputs, ids=all_ids, other_fields={})
        return dataset
    
    def get_dataset_info(self):
        """Get information about this MMLU dataset."""
        info = super().get_dataset_info()
        info.update({
            'subcategories': self.subcategories,
            'num_subcategories': len(self.subcategories),
            'split': self.split,
            'total_available_samples': sum(
                len(df) for df in self._cached_data.values()
            ) if self._cached_data else "Unknown (data not loaded)"
        })
        return info

# Example usage and testing
if __name__ == "__main__":
    print("Testing MMLUPromptDataset...")
    
    # Test with a few subcategories
    test_categories = ["college_biology", "high_school_mathematics", "anatomy"]
    
    # Create dataset instance
    mmlu_dataset = MMLUDataset(
        subcategories=test_categories,
        split="test"
    )
    
    print(f"\nDataset info: {mmlu_dataset.get_dataset_info()}")
    
    # Test data generation
    try:
        results = mmlu_dataset.generate_data(
            train_samples=30,  # 10 per category
            test_samples=30,   # 10 per category  
            save_dir="/tmp/mmlu_test",
            train_skip=0,
            test_skip=50  # Use different data for test
        )
        
        print("\n✓ Generated datasets:")
        print(f"  Train: {len(results['train'])} samples")
        print(f"  Test: {len(results['test'])} samples")
        
        # Show example question
        example_question = results['train'].inputs[0][0].content
        print(f"\nExample question:\n{example_question[:300]}...")
        
    except Exception as e:
        print(f"Error during generation: {e}")
    
    # Test single category dataset
    print("\n" + "="*50)
    print("Testing single category MMLU dataset...")
    
    # Use the main class with a single subcategory
    single_dataset = MMLUDataset(
        subcategories=["college_biology"],
        split="test"
    )
    print(f"Single dataset info: {single_dataset.get_dataset_info()}")
