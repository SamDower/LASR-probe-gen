"""
Centralized imports for all dataset classes.

This module provides a single import location for all PromptDataset subclasses,
making it easy to import any dataset class without needing to know the specific
module name.

Usage:
    from probe_gen.annotation.datasets import ScienceDataset, UltrachatDataset, ShakespeareDataset
    
    # Or import all
    from probe_gen.annotation.datasets import *
"""

# Base classes
from .authority_arguments_dataset import AuthorityArgumentsDataset
from .authority_multichoice_dataset import AuthorityMultichoiceDataset
from .coding_dataset import CodingQuestionsDataset
from .interface_dataset import Dataset, Message
from .jailbreak_dataset import HarmfulRequestDataset, JailbreakDataset
from .mmlu_dataset import MMLUDataset
from .prompt_dataset import PromptDataset

# Specific dataset implementations
from .refusal_dataset import RefusalDataset
from .sandbagging_multi_dataset import SandbaggingMultiDataset
from .shakespeare_dataset import ShakespeareDataset
from .sycophancy_arguments_dataset import SycophancyArgumentsDataset
from .sycophancy_multichoice_dataset import SycophancyMultichoiceDataset
from .sycophancy_poems_dataset import SycophancyPoemsDataset
from .tinystories_dataset import TinyStoriesDataset
from .ultrachat_dataset import UltrachatDataset
from .writingprompts_dataset import WritingPromptsDataset

# Define what gets imported with "from datasets import *"
__all__ = [
    # Base classes
    'PromptDataset',
    'Dataset', 
    'Message',
    
    # Dataset implementations
    'JailbreakDataset',
    'HarmfulRequestDataset',
    'RefusalDataset',
    'MMLUDataset',
    'UltrachatDataset', 
    'SandbaggingMultiDataset',
    'ShakespeareDataset',
    'SycophancyArgumentsDataset',
    'SycophancyMultichoiceDataset',
    'SycophancyPoemsDataset',
    'TinyStoriesDataset',
    'JailbreakDataset',
    'HarmfulRequestDataset',
    'AuthorityArgumentsDataset',
    'AuthorityMultichoiceDataset',
    'WritingPromptsDataset',
    'CodingQuestionsDataset',
]
