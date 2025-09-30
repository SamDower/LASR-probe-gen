"""
Ultrachat dataset implementation as a PromptDataset subclass.

This module provides a clean interface for generating Ultrachat datasets 
with automatic train/test separation and consistent interface. Supports both
regular Ultrachat and Brazilian Ultrachat datasets.
"""

from collections import defaultdict
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

from probe_gen.annotation.interface_dataset import Dataset, Message
from probe_gen.annotation.prompt_dataset import PromptDataset

# System prompts for different analysis types (from original ultrachat_behaviour.py)
SYSTEM_PROMPT_LISTS = """\
Please analyze whether the given dialogue between user and assistant involves the assistant including a list in their response.

Respond with a JSON using the following keys:
- reason: your reasoning and a short explanation for your answer.
- answer: a number between 1 and 10 indicating, where 1 means the assistant definitely included a list in their response and 10 means the assistant definitely did not include a list in their response.
- confidence: a number between 1 and 10 indicating your confidence in your answer.
"""

SYSTEM_PROMPT_FORMALITY = """\
Please analyze whether the given dialogue between user and assistant involves the assistant responding formally and seriously in their response.

Respond with a JSON using the following keys:
- reason: your reasoning and a short explanation for your answer.
- answer: a number between 1 and 10 indicating, where 1 means the assistant definitely was formal and serious in its response and 10 means the assistant definitely was casual and light-hearted in its response.
- confidence: a number between 1 and 10 indicating your confidence in your answer.
"""

SYSTEM_PROMPT_METAPHORS = """\
Please analyze whether the given dialogue between user and assistant involves the assistant responding with a metaphor or imagery in their response.

Respond with a JSON using the following keys:
- reason: your reasoning and a short explanation for your answer.
- answer: a number between 1 and 10 indicating, where 1 means the assistant definitely was using a metaphor or imagery in its response and 10 means the assistant definitely was not using a metaphor or imagery in its response.
- confidence: a number between 1 and 10 indicating your confidence in your answer.
"""

SYSTEM_PROMPT_SCIENCE = """\
Please analyze whether the given dialogue between user and assistant involves the assistant responding with reference to scientific facts.

Respond with a JSON using the following keys:
- reason: your reasoning and a short explanation for your answer.
- answer: a number between 1 and 10 indicating, where 1 means the assistant definitely was saying scientific facts in its response and 10 means the assistant definitely was not talking about scientific facts in its response.
- confidence: a number between 1 and 10 indicating your confidence in your answer.
"""


class UltrachatDataset(PromptDataset):
    """
    Ultrachat dataset implementation with unified PromptDataset interface.
    
    This class handles both regular Ultrachat and Brazilian Ultrachat datasets
    with automatic data streaming from HuggingFace and proper train/test separation.
    
    Features:
    - Supports both "ultrachat" and "ultrachat_brazilian" variants
    - Automatic prompt filtering by length
    - Proper train/test separation using skip parameters
    - Streaming dataset support for large datasets
    - Consistent interface with other prompt datasets
    """
    
    def __init__(
        self, 
        variant: str = "ultrachat",
        max_prompt_length: int = 500,
        default_train_test_gap: int = 40000
    ):
        """
        Initialize Ultrachat prompt dataset.
        
        Args:
            variant: Either "ultrachat" or "ultrachat_brazilian"
            max_prompt_length: Maximum prompt length to include
            default_train_test_gap: Default gap between train and test data
        """
        super().__init__(f"ultrachat_{variant}", default_train_test_gap)
        
        if variant not in ["ultrachat", "ultrachat_brazilian"]:
            raise ValueError(f"Invalid variant: {variant}. Must be 'ultrachat' or 'ultrachat_brazilian'")
        
        self.variant = variant
        self.max_prompt_length = max_prompt_length
        self._cached_samples = None  # Cache processed samples to avoid reprocessing
        
        # Configure dataset parameters based on variant
        if variant == "ultrachat":
            self.hf_dataset_name = "HuggingFaceH4/ultrachat_200k"
            self.hf_split = "train_sft"
        else:  # ultrachat_brazilian
            self.hf_dataset_name = "recogna-nlp/UltrachatBR"
            self.hf_split = "train"
    
    def download_data(self) -> None:
        """
        Download and cache Ultrachat data from HuggingFace.
        
        The data is processed and cached in memory to avoid repeated processing.
        Due to the large size of these datasets, we use streaming mode.
        """
        print(f"Loading {self.variant} data from HuggingFace...")
        
        # We don't actually download the full dataset here since it's huge
        # Instead, we'll stream it during dataset creation
        print(f"✓ Ready to stream {self.variant} data from {self.hf_dataset_name}")
    
    def _process_ultrachat_sample(self, sample, real_ix: int) -> Optional[tuple]:
        """
        Process a regular Ultrachat sample.
        
        Args:
            sample: Raw sample from HuggingFace dataset
            real_ix: Real index in the dataset
            
        Returns:
            Tuple of (messages, sample_id) or None if sample should be skipped
        """
        prompt = sample['prompt']
        
        # Skip samples with long inputs
        if len(prompt) > self.max_prompt_length:
            return None
        
        messages = [
            Message(role="user", content=prompt), 
            Message(role="assistant", content="")
        ]
        sample_id = f"{self.hf_split}_{real_ix}_prompt"
        
        return (messages, sample_id)
    
    def _process_brazilian_sample(self, sample, real_ix: int) -> Optional[tuple]:
        """
        Process a Brazilian Ultrachat sample.
        
        Args:
            sample: Raw sample from HuggingFace dataset
            real_ix: Real index in the dataset
            
        Returns:
            Tuple of (messages, sample_id) or None if sample should be skipped
        """
        # Parse the Brazilian conversation format
        conversa = sample['conversa']
        
        if "[{'humano': '" not in conversa or "', 'assistente': '" not in conversa:
            return None
        
        try:
            # Extract human and assistant messages
            prompt = conversa.split("[{'humano': '")[1]
            human_message = prompt.split("', 'assistente': '")[0]
            prompt = prompt.split("', 'assistente': '")[1]
            assistant_message = prompt.split("'}")[0]
            
            # Skip samples with long inputs
            if len(human_message) >= self.max_prompt_length:
                return None
            
            messages = [
                Message(role="user", content=human_message), 
                Message(role="assistant", content=assistant_message)
            ]
            sample_id = f"{self.hf_split}_{real_ix}_conversa"
            
            return (messages, sample_id)
        
        except (IndexError, ValueError):
            # Skip malformed samples
            return None
    
    def _create_dataset(self, mode: str, n_samples: int, skip: int) -> Dataset:
        """
        Create Ultrachat dataset with specified parameters using streaming.
        
        Args:
            mode: "train" or "test" (used for identification only)
            n_samples: Number of samples to generate
            skip: Number of samples to skip from the beginning
            
        Returns:
            Dataset object with Ultrachat conversations
            
        Raises:
            ValueError: If not enough samples are available
        """
        print(f"Streaming {self.variant} data to create {n_samples} samples (skip={skip})...")
        
        # Stream the dataset
        hf_dataset = load_dataset(
            self.hf_dataset_name, 
            split=self.hf_split, 
            streaming=True
        )
        
        ids = []
        inputs = []
        other_fields = defaultdict(list)
        
        processed_count = 0  # Count of valid (non-skipped) samples
        
        for real_ix, sample in tqdm(enumerate(hf_dataset), desc=f"Processing {self.variant}"):
            # Process the sample based on variant
            if self.variant == "ultrachat":
                result = self._process_ultrachat_sample(sample, real_ix)
            else:  # ultrachat_brazilian
                result = self._process_brazilian_sample(sample, real_ix)
            
            # Skip invalid samples
            if result is None:
                continue
            
            messages, sample_id = result
            
            # Check if we're past the skip threshold
            if processed_count >= skip:
                # Check we're not past the train/test gap
                if mode == "test" or processed_count <= self.default_train_test_gap:
                    # Update sample ID to include the mode
                    updated_sample_id = f"{sample_id}_{mode}"
                    
                    ids.append(updated_sample_id)
                    inputs.append(messages)
                    
                    # Stop if we have enough samples
                    if len(ids) >= n_samples:
                        break
            
            processed_count += 1
        
        # Check if we have enough samples
        if len(ids) < n_samples:
            print("Didn't find enough samples, returning the remainder")
        
        dataset = Dataset(inputs=inputs, ids=ids, other_fields=other_fields)
        return dataset
    
    def get_dataset_info(self):
        """Get information about this Ultrachat dataset."""
        info = super().get_dataset_info()
        info.update({
            'variant': self.variant,
            'hf_dataset_name': self.hf_dataset_name,
            'hf_split': self.hf_split,
            'max_prompt_length': self.max_prompt_length,
            'description': f"{self.variant.replace('_', ' ').title()} dataset for generating conversational prompts"
        })
        return info


# Example usage and testing
if __name__ == "__main__":
    print("Testing UltrachatDataset...")
    
    # Test regular Ultrachat
    print("\n" + "="*50)
    print("Testing regular Ultrachat dataset...")
    
    ultrachat_dataset = UltrachatDataset(
        variant="ultrachat",
        max_prompt_length=300
    )
    
    print(f"\nDataset info: {ultrachat_dataset.get_dataset_info()}")
    
    # Test data generation
    try:
        results = ultrachat_dataset.generate_data(
            train_samples=10,
            test_samples=5,
            save_dir="/tmp/ultrachat_test"
        )
        
        print("\n✓ Generated datasets:")
        print(f"  Train: {len(results['train'])} samples")
        print(f"  Test: {len(results['test'])} samples")
        
        # Show example question
        example_prompt = results['train'].inputs[0][0].content
        print(f"\nExample prompt:\n{example_prompt[:200]}...")
        
    except Exception as e:
        print(f"Error during generation: {e}")
    
    # Test Brazilian Ultrachat
    print("\n" + "="*50)
    print("Testing Brazilian Ultrachat dataset...")
    
    brazilian_dataset = UltrachatDataset(
        variant="ultrachat_brazilian",
        max_prompt_length=300
    )
    
    print(f"Brazilian dataset info: {brazilian_dataset.get_dataset_info()}")
    
    # Test single dataset mode
    try:
        results = brazilian_dataset.generate_data(
            train_samples=5,
            save_dir="/tmp/ultrachat_brazilian_test"
        )
        
        print(f"\n✓ Generated Brazilian train dataset: {len(results['train'])} samples")
        
        # Show example conversation
        example_messages = results['train'].inputs[0]
        print("\nExample Brazilian conversation:")
        print(f"User: {example_messages[0].content[:100]}...")
        print(f"Assistant: {example_messages[1].content[:100]}...")
        
    except Exception as e:
        print(f"Error during Brazilian generation: {e}")
