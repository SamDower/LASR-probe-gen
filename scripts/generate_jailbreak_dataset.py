#!/usr/bin/env python3
"""
Script to generate a 100k mixed jailbreak dataset and save to CSV.
This combines original jailbreak data with generated harmful request wrappers.
"""

import argparse
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe_gen.annotation.jailbreak_dataset import JailbreakDataset
from probe_gen.annotation.interface_dataset import Message

def generate_mixed_jailbreak_dataset(n_samples=100000, output_file="data/jailbreak_mixed_100k.csv"):
    """
    Generate a mixed jailbreak dataset with original jailbreak samples and wrapped harmful requests.
    
    Args:
        n_samples: Total number of samples to generate (default: 100,000)
        output_file: Path to save the CSV file
    """
    print(f"Generating {n_samples:,} mixed jailbreak samples...")
    
    # Create temp directory
    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the jailbreak dataset
    jailbreak_ds = JailbreakDataset()
    jailbreak_ds.download_data()
    
    # Calculate split: 50% original jailbreak, 50% wrapped harmful requests
    # But limit based on available data
    max_available = 10200  # Based on the jailbreak dataset size
    original_samples = min(n_samples // 2, max_available)
    wrapped_samples = n_samples - original_samples
    
    print(f"Adjusted split: {original_samples:,} original, {wrapped_samples:,} wrapped")
    
    print(f"Generating {original_samples:,} original jailbreak samples...")
    print(f"Generating {wrapped_samples:,} wrapped harmful request samples...")
    
    all_data = []
    
    # 1. Generate original jailbreak samples
    if original_samples > 0:
        print("Generating original jailbreak samples...")
        original_success = jailbreak_ds.generate_data(
            mode="train",
            n_samples=original_samples,
            skip=0,
            output_file="data/temp/original_jailbreak.jsonl"
        )
        
        if original_success:
            # Load the generated dataset from the file
            from probe_gen.annotation.interface_dataset import Dataset
            original_dataset = Dataset.load_from("data/temp/original_jailbreak.jsonl")
            
            for i, (input_msgs, sample_id, metadata) in enumerate(zip(
                original_dataset.inputs,
                original_dataset.ids,
                zip(*[original_dataset.other_fields[field] for field in original_dataset.other_fields.keys()])
            )):
                # Extract user message content
                user_content = input_msgs[0].content if input_msgs else ""
                
                # Create metadata dict
                metadata_dict = dict(zip(original_dataset.other_fields.keys(), metadata))
                
                all_data.append({
                    'id': sample_id,
                    'user_content': user_content,
                    'assistant_content': '',  # Empty for generation
                    'prompt_name': metadata_dict.get('prompt_name', ''),
                    'jailbreak_prompt_name': metadata_dict.get('jailbreak_prompt_name', ''),
                    'original_prompt_text': metadata_dict.get('original_prompt_text', ''),
                    'jailbreak_prompt_text': metadata_dict.get('jailbreak_prompt_text', ''),
                    'sample_type': 'original_jailbreak',
                    'shuffled_index': metadata_dict.get('shuffled_index', i),
                    'processed_index': metadata_dict.get('processed_index', i)
                })
        else:
            print("Warning: Failed to generate original jailbreak samples")
    
    # 2. Generate wrapped harmful request samples
    if wrapped_samples > 0:
        print("Generating wrapped harmful request samples...")
        wrapped_success = jailbreak_ds.generate_data(
            mode="train", 
            n_samples=wrapped_samples,
            skip=0,  # Start from beginning for wrapped samples
            output_file="data/temp/wrapped_harmful.jsonl"
        )
        
        if wrapped_success:
            # Load the generated dataset from the file
            from probe_gen.annotation.interface_dataset import Dataset
            wrapped_dataset = Dataset.load_from("data/temp/wrapped_harmful.jsonl")
            
            for i, (input_msgs, sample_id, metadata) in enumerate(zip(
                wrapped_dataset.inputs,
                wrapped_dataset.ids,
                zip(*[wrapped_dataset.other_fields[field] for field in wrapped_dataset.other_fields.keys()])
            )):
                # Extract user message content
                user_content = input_msgs[0].content if input_msgs else ""
                
                # Create metadata dict
                metadata_dict = dict(zip(wrapped_dataset.other_fields.keys(), metadata))
                
                all_data.append({
                    'id': sample_id,
                    'user_content': user_content,
                    'assistant_content': '',  # Empty for generation
                    'prompt_name': metadata_dict.get('prompt_name', ''),
                    'jailbreak_prompt_name': metadata_dict.get('jailbreak_prompt_name', ''),
                    'original_prompt_text': metadata_dict.get('original_prompt_text', ''),
                    'jailbreak_prompt_text': metadata_dict.get('jailbreak_prompt_text', ''),
                    'sample_type': 'wrapped_harmful',
                    'shuffled_index': metadata_dict.get('shuffled_index', i),
                    'processed_index': metadata_dict.get('processed_index', i)
                })
        else:
            print("Warning: Failed to generate wrapped harmful samples")
    
    # 3. Create DataFrame and shuffle
    print("Creating DataFrame and shuffling...")
    df = pd.DataFrame(all_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 4. Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {len(df):,} samples to {output_path}...")
    df.to_csv(output_path, index=False)
    
    # 5. Print summary
    print(f"\n✓ Dataset generated successfully!")
    print(f"Total samples: {len(df):,}")
    print(f"Original jailbreak: {len(df[df['sample_type'] == 'original_jailbreak']):,}")
    print(f"Wrapped harmful: {len(df[df['sample_type'] == 'wrapped_harmful']):,}")
    print(f"Saved to: {output_path}")
    
    # Print sample types breakdown
    print(f"\nSample type breakdown:")
    print(df['sample_type'].value_counts())
    
    # Print jailbreak prompt name breakdown
    print(f"\nJailbreak prompt name breakdown:")
    print(df['jailbreak_prompt_name'].value_counts())
    
    # Clean up temp files
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"\n✓ Cleaned up temporary files")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate mixed jailbreak dataset")
    parser.add_argument("--n_samples", type=int, default=100000, help="Total number of samples")
    parser.add_argument("--output_file", type=str, default="data/jailbreak_mixed_100k.csv", help="Output CSV file path")
    
    args = parser.parse_args()
    
    generate_mixed_jailbreak_dataset(
        n_samples=args.n_samples,
        output_file=args.output_file
    )

if __name__ == "__main__":
    main()
