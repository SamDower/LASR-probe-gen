"""
TinyStories dataset implementation as a PromptDataset subclass.

This module provides a clean interface for generating TinyStories datasets
with automatic train/test separation and consistent interface. Uses the
roneneldan/TinyStories dataset for creating story generation prompts.
"""

from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm

from probe_gen.annotation.interface_dataset import Dataset, Message
from probe_gen.annotation.prompt_dataset import PromptDataset

# System prompts for different analysis types (from original tinystories_behaviour.py)
SYSTEM_PROMPT_REFUSAL_STORY = """\
Please analyze whether the given story involves one character rejecting the other in some way.

Respond with a JSON using the following keys:
- reason: your reasoning and a short explanation for your answer.
- answer: a number between 1 and 10 indicating, where 1 means the story definitely involves one character rejecting the other and 10 means the story definitely does not involve one character rejecting the other.
- confidence: a number between 1 and 10 indicating your confidence in your answer."""

# pos:neg ratio is like 1:11
SYSTEM_PROMPT_LISTS_STORY = """\
Please analyze whether the given story involves one character listing something.

Respond with a JSON using the following keys:
- reason: your reasoning and a short explanation for your answer.
- answer: a number between 1 and 10 indicating, where 1 means the story definitely involves one character listing something and 10 means the story definitely does not involve one character listing something.
- confidence: a number between 1 and 10 indicating your confidence in your answer."""

# pos:neg ratio is like 1:8 but we probably dont want this test
SYSTEM_PROMPT_METAPHOR_STORY = """\
Please analyze whether the given story involves one character using a metaphor or imagery in their response.

Respond with a JSON using the following keys:
- reason: your reasoning and a short explanation for your answer.
- answer: a number between 1 and 10 indicating, where 1 means the story definitely involves one character using a metaphor or imagery in their response and 10 means the story definitely does not involve one character using a metaphor or imagery in their response.
- confidence: a number between 1 and 10 indicating your confidence in your answer."""

# pos:neg ratio is like 1:20 
SYSTEM_PROMPT_SCIENCE_STORY = """\
Please analyze whether the given story involves one character saying something with reference to scientific facts.

Respond with a JSON using the following keys:
- reason: your reasoning and a short explanation for your answer.
- answer: a number between 1 and 10 indicating, where 1 means the story definitely involves one character saying something with reference to scientific facts or knowledge and 10 means the story definitely does not involve one character saying something with reference to scientific facts or knowledge.
- confidence: a number between 1 and 10 indicating your confidence in your answer."""


class TinyStoriesDataset(PromptDataset):
    """
    TinyStories dataset implementation with unified PromptDataset interface.
    
    This class handles the roneneldan/TinyStories dataset with automatic
    data streaming from HuggingFace and proper train/test separation.
    Creates datasets for story generation and analysis.
    
    Features:
    - Streams data from roneneldan/TinyStories HuggingFace dataset
    - Filters stories by maximum length
    - Creates story telling conversations
    - Proper train/test separation using skip parameters
    - Consistent interface with other prompt datasets
    """
    
    def __init__(
        self, 
        hf_dataset_name: str = "roneneldan/TinyStories",
        split: str = "train",
        max_story_length: int = 1000,
        story_prompt: str = "Tell me a story",
        default_train_test_gap: int = 1000
    ):
        """
        Initialize TinyStories prompt dataset.
        
        Args:
            hf_dataset_name: HuggingFace dataset identifier
            split: Which dataset split to use ("train", "validation")
            max_story_length: Maximum length for stories to include
            story_prompt: The prompt to use for requesting stories
            default_train_test_gap: Default gap between train and test data
        """
        super().__init__("tinystories", default_train_test_gap)
        
        self.hf_dataset_name = hf_dataset_name
        self.split = split
        self.max_story_length = max_story_length
        self.story_prompt = story_prompt
        self._cached_samples = None  # Cache processed samples to avoid reprocessing
    
    def download_data(self) -> None:
        """
        Download and cache TinyStories data from HuggingFace.
        
        The data is streamed rather than fully downloaded due to size.
        """
        print(f"Loading TinyStories data from HuggingFace: {self.hf_dataset_name}")
        
        # We don't actually download the full dataset here since we use streaming
        # Just verify that the dataset is accessible
        try:
            # Test loading a small sample to verify dataset availability
            test_dataset = load_dataset(self.hf_dataset_name, split=self.split, streaming=True)
            # Try to get the first sample to verify access
            next(iter(test_dataset))
            print(f"✓ Successfully verified access to {self.hf_dataset_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to access HuggingFace dataset: {e}")
    
    def _create_dataset(self, mode: str, n_samples: int, skip: int) -> Dataset:
        """
        Create TinyStories dataset with specified parameters using streaming.
        
        Args:
            mode: "train" or "test" (used for identification only)
            n_samples: Number of samples to generate
            skip: Number of samples to skip from the beginning
            
        Returns:
            Dataset object with story generation conversations
            
        Raises:
            ValueError: If not enough samples are available
        """
        print(f"Streaming {self.hf_dataset_name} to create {n_samples} samples (skip={skip})...")
        
        # Stream the HuggingFace dataset
        hf_dataset = load_dataset(self.hf_dataset_name, split=self.split, streaming=True)
        
        ids = []
        inputs = []
        other_fields = defaultdict(list)
        
        processed_count = 0  # Count of valid (non-skipped) samples
        
        for real_ix, sample in tqdm(enumerate(hf_dataset), desc=f"Processing {self.hf_dataset_name}"):
            story_text = sample["text"]
            
            # Skip samples with long stories
            if len(story_text) > self.max_story_length:
                continue
            
            # Check if we're past the skip threshold
            if processed_count >= skip:
                # Check we're not past the train/test gap
                if mode == "test" or processed_count <= self.default_train_test_gap:
                    # Create conversation with user request and assistant story
                    messages = [
                        Message(role="user", content=self.story_prompt),
                        Message(role="assistant", content=story_text),
                    ]
                    
                    # Create unique sample ID
                    sample_id = f"{self.split}_{real_ix}_text_{mode}"
                    
                    ids.append(sample_id)
                    inputs.append(messages)
                    
                    # Add metadata
                    other_fields["original_index"].append(real_ix)
                    other_fields["processed_index"].append(processed_count)
                    other_fields["story_length"].append(len(story_text))
                    other_fields["story_preview"].append(story_text[:100] + "..." if len(story_text) > 100 else story_text)
                    
                    # Stop if we have enough samples
                    if len(ids) >= n_samples:
                        break
            
            processed_count += 1
        
        # Check if we have enough samples
        if len(ids) < n_samples:
            raise ValueError(
                f"Not enough samples found in the dataset. Found {len(ids)}, expected {n_samples}. "
                f"Total processed samples: {processed_count}"
            )
        
        dataset = Dataset(inputs=inputs, ids=ids, other_fields=other_fields)
        return dataset
    
    def get_dataset_info(self):
        """Get information about this TinyStories dataset."""
        info = super().get_dataset_info()
        info.update({
            'hf_dataset_name': self.hf_dataset_name,
            'split': self.split,
            'max_story_length': self.max_story_length,
            'story_prompt': self.story_prompt,
            'description': "TinyStories dataset for story generation and analysis",
            'filtering': f"Only includes stories with length <= {self.max_story_length} characters"
        })
        return info


# Legacy function for backward compatibility
def create_tinystories_dataset(num_samples: int = 1000, skip: int = 0) -> Dataset:
    """
    Create TinyStories dataset using the legacy function interface.
    
    This function is kept for backward compatibility.
    New code should use TinyStoriesDataset class directly.
    
    Args:
        num_samples: Number of samples to generate
        skip: Number of samples to skip from the beginning
        
    Returns:
        Dataset object with story conversations
    """
    dataset_generator = TinyStoriesDataset()
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
    print("Testing TinyStoriesDataset...")
    
    # Test dataset initialization
    print("\n" + "="*50)
    print("Testing TinyStories dataset...")
    
    tinystories_dataset = TinyStoriesDataset(
        max_story_length=500  # Lower for testing
    )
    
    print(f"\nDataset info: {tinystories_dataset.get_dataset_info()}")
    
    # Test data download verification
    try:
        print("\nTesting data access...")
        tinystories_dataset.download_data()
        print("✓ Data access verification successful")
        
        # Test small dataset generation
        print("\nTesting dataset generation...")
        results = tinystories_dataset.generate_data(
            train_samples=5,
            test_samples=3,
            save_dir="/tmp/tinystories_test"
        )
        
        print("\n✓ Generated datasets:")
        print(f"  Train: {len(results['train'])} samples")
        print(f"  Test: {len(results['test'])} samples")
        
        # Show example story
        example_messages = results['train'].inputs[0]
        print("\nExample conversation:")
        print(f"  User: {example_messages[0].content}")
        print(f"  Assistant: {example_messages[1].content[:100]}...")
        
        # Show some metadata
        if results['train'].other_fields:
            print("\nExample metadata:")
            print(f"  Original index: {results['train'].other_fields['original_index'][0]}")
            print(f"  Story length: {results['train'].other_fields['story_length'][0]}")
            print(f"  Story preview: {results['train'].other_fields['story_preview'][0]}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Note: This might fail if HuggingFace dataset is not accessible")
    
    # Test legacy function compatibility
    print("\n" + "="*50)
    print("Testing legacy function compatibility...")
    
    try:
        legacy_dataset = create_tinystories_dataset(num_samples=3, skip=0)
        print(f"\n✓ Legacy function works: {len(legacy_dataset)} samples")
        
        # Show example from legacy function
        legacy_example = legacy_dataset.inputs[0]
        print(f"Legacy example: {legacy_example[0].content} -> {legacy_example[1].content[:50]}...")
        
    except Exception as e:
        print(f"Error during legacy testing: {e}")
    
    # Test custom story prompt
    print("\n" + "="*30)
    print("Testing custom story prompt...")
    
    custom_dataset = TinyStoriesDataset(
        story_prompt="Write a short children's story",
        max_story_length=300
    )
    
    print(f"Custom dataset info: {custom_dataset.get_dataset_info()}")
    
    print("\n" + "="*50)
    print("TinyStories dataset testing complete!")
    print("\nNote: Full functionality requires:")
    print("1. Internet connection for accessing HuggingFace dataset")
    print("2. HuggingFace datasets library installed")
