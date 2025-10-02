from collections import defaultdict

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from probe_gen.annotation.interface_dataset import Dataset, Message


# 5 diverse MMLU subcategories covering different domains
MMLU_SUBCATEGORIES = [
    "college_biology",           # Life Sciences
    "college_computer_science",  # Computer Science
    "high_school_mathematics",   # Mathematics
    "anatomy",                   # Medical/Health Sciences
    "human_sexuality",            # Human Sexuality
    "high_school_psychology",     # Psychology
    "business_ethics",            # Social Sciences/Ethics
    "logical_fallacies",           # Logical Fallacies
    "high_school_world_history",        # History
    "high_school_us_history",          # History
    "management",
    "marketing", 
    "sociology", 
    "world_religions",
]


def create_mmlu_dataset(
    subcategory: str, 
    num_samples: int = 1000, 
    skip: int = 0, 
    split: str = "test"
) -> Dataset:
    """
    Create an MMLU dataset for a specific subcategory in the same format as the jailbreak dataset.
    
    Args:
        subcategory: MMLU subcategory (e.g., "college_biology", "anatomy", etc.)
        num_samples: Number of samples to include in the dataset
        skip: Number of samples to skip from the beginning
        split: Which split to use ("test", "dev", "val")
        
    Returns:
        Dataset object with MMLU questions formatted as conversations
    """
    # Load the specific MMLU subcategory
    hf_dataset = load_dataset("cais/mmlu", subcategory, split=split)
    df = pd.DataFrame(hf_dataset)
    
    ids = []
    inputs = []
    
    # Apply skip and limit
    df_subset = df.iloc[skip:skip + num_samples]
    
    if len(df_subset) < num_samples:
        print(f"Warning: Only {len(df_subset)} samples available for {subcategory} after skip={skip}, requested {num_samples}")
    
    for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc=f"Processing MMLU {subcategory}"):
        # Format the question with multiple choice options
        question = row['question']
        choices = row['choices']
        
        # Create the formatted question with choices (A, B, C, D)
        formatted_question = f"{question}\n\n"
        for i, choice_text in enumerate(choices):
            choice_label = chr(65 + i)  # A, B, C, D
            formatted_question += f"{choice_label}. {choice_text}\n"
        
        # Create conversation messages with user question and empty assistant response
        messages = [
            Message(role="user", content=formatted_question.strip()),
            Message(role="assistant", content="")
        ]
        
        # Create a unique ID
        sample_id = f"mmlu_{subcategory}_{idx}"
        
        ids.append(sample_id)
        inputs.append(messages)
    
    dataset = Dataset(inputs=inputs, ids=ids, other_fields={})
    return dataset


def create_all_mmlu_subcategories(
    num_samples_per_category: int = 200, 
    skip: int = 0, 
    split: str = "test"
) -> dict:
    """
    Create datasets for all 5 MMLU subcategories.
    
    Args:
        num_samples_per_category: Number of samples per subcategory
        skip: Number of samples to skip from the beginning
        split: Which split to use
        
    Returns:
        Dictionary mapping subcategory names to Dataset objects
    """
    datasets = {}
    
    for subcategory in MMLU_SUBCATEGORIES:
        print(f"\nProcessing {subcategory}...")
        datasets[subcategory] = create_mmlu_dataset(
            subcategory=subcategory,
            num_samples=num_samples_per_category,
            skip=skip,
            split=split
        )
    
    return datasets


def create_combined_mmlu_dataset(
    num_samples_per_category: int = 200, 
    skip: int = 0, 
    split: str = "test"
) -> Dataset:
    """
    Create a single combined dataset with samples from all 5 subcategories.
    
    Args:
        num_samples_per_category: Number of samples per subcategory
        skip: Number of samples to skip from the beginning  
        split: Which split to use
        
    Returns:
        Single Dataset object with samples from all subcategories
    """
    all_ids = []
    all_inputs = []
    
    for subcategory in MMLU_SUBCATEGORIES:
        print(f"\nProcessing {subcategory} for combined dataset...")
        
        # Load the specific MMLU subcategory
        hf_dataset = load_dataset("cais/mmlu", subcategory, split=split)
        df = pd.DataFrame(hf_dataset)
        
        # Apply skip and limit
        df_subset = df.iloc[skip:skip + num_samples_per_category]
        
        if len(df_subset) < num_samples_per_category:
            print(f"Warning: Only {len(df_subset)} samples available for {subcategory} after skip={skip}, requested {num_samples_per_category}")
        
        for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc=f"Processing {subcategory}"):
            # Format the question with multiple choice options
            question = row['question']
            choices = row['choices']
            
            # Create the formatted question with choices (A, B, C, D)
            formatted_question = f"{question}\n\n"
            for i, choice_text in enumerate(choices):
                choice_label = chr(65 + i)  # A, B, C, D
                formatted_question += f"{choice_label}. {choice_text}\n"
            
            # Create conversation messages with user question and empty assistant response
            messages = [
                Message(role="user", content=formatted_question.strip()),
                Message(role="assistant", content="")
            ]
            
            # Create a unique ID that includes the subcategory
            sample_id = f"mmlu_{subcategory}_{idx}"
            
            all_ids.append(sample_id)
            all_inputs.append(messages)
    
    dataset = Dataset(inputs=all_inputs, ids=all_ids, other_fields={})
    return dataset


# Example usage:
if __name__ == "__main__":
    print("Available MMLU subcategories:")
    for i, category in enumerate(MMLU_SUBCATEGORIES, 1):
        print(f"{i}. {category}")
    
    # Create individual datasets for each subcategory
    individual_datasets = create_all_mmlu_subcategories(num_samples_per_category=100)
    
    for category, dataset in individual_datasets.items():
        print(f"\n{category}: {len(dataset.inputs)} samples")
        print("Example question:")
        print(dataset.inputs[0][0].content[:200] + "...")
    
    # Create a combined dataset
    print(f"\n" + "="*50)
    print("Creating combined dataset...")
    combined_dataset = create_combined_mmlu_dataset(num_samples_per_category=50)
    print(f"\nCombined dataset: {len(combined_dataset.inputs)} total samples")
    
    # Show distribution by category
    category_counts = {}
    for sample_id in combined_dataset.ids:
        category = sample_id.split('_')[1]  # Extract category from ID
        category_counts[category] = category_counts.get(category, 0) + 1
    
    print("\nSamples per category in combined dataset:")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")