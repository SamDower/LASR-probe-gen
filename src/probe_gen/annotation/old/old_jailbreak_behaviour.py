from collections import defaultdict

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from probe_gen.annotation.interface_dataset import Dataset, Message

# Key design decision: The jailbreak dataset preserves both the jailbreak_prompt_text 
# (which the model responds to) and the original_prompt_text (which the autograder 
# uses for evaluation). This allows the model to be tested on jailbreak attempts 
# while the autograder can still evaluate the response quality against the original prompt.

SYSTEM_PROMPT_JAILBREAK = """ """

# def create_jailbreak_dataset(num_samples: int = 1000, skip: int = 0) -> Dataset:
#     """
#     Create a jailbreak dataset in the same format as the refusal dataset.
    
#     Args:
#         num_samples: Number of samples to include in the dataset
#         skip: Number of samples to skip from the beginning
        
#     Returns:
#         Dataset object with jailbreak prompts formatted as conversations
#     """
#     # Load the jailbreak dataset
#     hf_dataset = load_dataset("sevdeawesome/jailbreak_success", split="train")
#     df = pd.DataFrame(hf_dataset)
    
#     df = df[~df['jailbreak_prompt_name'].isin(['gcg', 'autodan'])]
    
#     ids = []
#     inputs = []
#     other_fields = defaultdict(list)
    
    
#     # Apply skip and limit
#     df_subset = df.iloc[skip:skip + num_samples]
    
#     if len(df_subset) < num_samples:
#         print(f"Warning: Only {len(df_subset)} samples available after skip={skip}, requested {num_samples}")
    
#     for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Processing jailbreak dataset"):
#         # Create conversation messages
#         # For jailbreak attempts, we typically have a user prompt and expect an assistant response
#         messages = []
        
#         # Use the jailbreak_prompt_text as the user message (this contains the jailbreak attempt)
#         user_content = row['jailbreak_prompt_text']
#         messages.append(Message(role="user", content=user_content))
        
#         # Create a unique ID
#         prompt_name = row['prompt_name']
#         jailbreak_type = row['jailbreak_prompt_name']
#         sample_id = f"jailbreak_{idx}_{prompt_name}_{jailbreak_type}"
        
#         ids.append(sample_id)
#         inputs.append(messages)
        
#         # Store additional fields that might be useful
#         other_fields['prompt_name'].append(prompt_name)
#         other_fields['jailbreak_prompt_name'].append(jailbreak_type)
#         other_fields['original_prompt_text'].append(row['original_prompt_text'])
#         other_fields['jailbreak_prompt_text'].append(row['jailbreak_prompt_text'])
        
#         # Add the original prompt text to 'input' field as requested
#         other_fields['input'].append(row['original_prompt_text'])
    
    
#     dataset = Dataset(inputs=inputs, ids=ids, other_fields=other_fields)
#     return dataset




# Key design decision: The jailbreak dataset preserves both the jailbreak_prompt_text
# (which the model responds to) and the original_prompt_text (which the autograder
# uses for evaluation). This allows the model to be tested on jailbreak attempts
# while the autograder can still evaluate the response quality against the original prompt.

SYSTEM_PROMPT_JAILBREAK = """ """

def create_jailbreak_dataset(num_samples: int = 1000, skip: int = 0) -> Dataset:
    """
    Create a jailbreak dataset in the same format as the MMLU dataset.
    
    Args:
        num_samples: Number of samples to include in the dataset
        skip: Number of samples to skip from the beginning
        
    Returns:
        Dataset object with jailbreak prompts formatted as conversations
    """
    # Load the jailbreak dataset
    hf_dataset = load_dataset("sevdeawesome/jailbreak_success", split="train")
    df = pd.DataFrame(hf_dataset)
    df = df[~df['jailbreak_prompt_name'].isin(['gcg', 'autodan'])]
    
    ids = []
    inputs = []
    
    # Apply skip and limit
    df_subset = df.iloc[skip:skip + num_samples]
    
    if len(df_subset) < num_samples:
        print(f"Warning: Only {len(df_subset)} samples available after skip={skip}, requested {num_samples}")
    
    for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Processing jailbreak dataset"):
        # Create conversation messages
        # Use the jailbreak_prompt_text as the user message (this contains the jailbreak attempt)
        user_content = row['jailbreak_prompt_text']
        
        # Create messages with user question and empty assistant response (matching MMLU structure)
        messages = [
            Message(role="user", content=user_content),
            Message(role="assistant", content="")
        ]
        
        # Create a unique ID
        prompt_name = row['prompt_name']
        jailbreak_type = row['jailbreak_prompt_name']
        sample_id = f"jailbreak_{idx}_{prompt_name}_{jailbreak_type}"
        
        ids.append(sample_id)
        inputs.append(messages)
    
    # Create dataset with minimal structure matching MMLU
    dataset = Dataset(inputs=inputs, ids=ids, other_fields={})
    
    return dataset


def create_harfmul_request_dataset(num_samples: int = 1000, skip: int = 0) -> Dataset:
    """
    Create a dataset from the vanilla_harmful prompts in allenai/wildjailbreak,
    in the same format as the jailbreak dataset.

    Args:
        num_samples: Number of samples to include
        skip: Number of samples to skip from the beginning

    Returns:
        Dataset object with vanilla_harmful prompts formatted as conversations
    """
    # Load the WildJailbreak dataset
    hf_dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)

    # Convert to DataFrame for easier filtering
    df = pd.DataFrame(hf_dataset['train'])  # or just hf_dataset if already split   
    df = df[df['data_type'] == 'vanilla_harmful']

    # Apply skip and limit
    df_subset = df.iloc[skip:skip + num_samples]

    if len(df_subset) < num_samples:
        print(f"Warning: Only {len(df_subset)} samples available after skip={skip}, requested {num_samples}")

    ids = []
    inputs = []
    other_fields = defaultdict(list)

    for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Processing harmful request dataset"):
        messages = [Message(role="user", content=row["vanilla"])]

        sample_id = f"harfmul_{idx}_{row.get('id', 'noid')}"
        ids.append(sample_id)
        inputs.append(messages)


        other_fields['input'].append(row['vanilla'])  # For consistency

    dataset = Dataset(inputs=inputs, ids=ids, other_fields=other_fields)
    return dataset



# # Alternative version if you want to include both original and jailbreak prompts as separate samples
# def create_jailbreak_dataset_with_originals(num_samples: int = 1000, skip: int = 0) -> Dataset:
#     """
#     Create a jailbreak dataset that includes both original prompts and jailbreak versions.
#     This doubles the dataset size by including both versions of each prompt.
#     """
#     # Load the jailbreak dataset
#     hf_dataset = load_dataset("sevdeawesome/jailbreak_success", split="train")
    
#     ids = []
#     inputs = []
#     other_fields = defaultdict(list)
    
#     # Convert to pandas for easier manipulation
#     df = pd.DataFrame(hf_dataset)
    
#     # Filter out rows with jailbreak_prompt_name == "gcg" or "autodan"
#     df = df[~df['jailbreak_prompt_name'].isin(['gcg', 'autodan'])]
    
#     # Calculate how many base samples we need (since we're creating 2 samples per row)
#     base_samples_needed = (num_samples + 1) // 2  # Round up
#     df_subset = df.iloc[skip:skip + base_samples_needed]
    
#     sample_count = 0
#     for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Processing jailbreak dataset"):
#         if sample_count >= num_samples:
#             break
            
#         prompt_name = row['prompt_name']
#         jailbreak_type = row['jailbreak_prompt_name']
        
#         # Add original prompt version
#         if sample_count < num_samples:
#             messages_orig = [Message(role="user", content=row['original_prompt_text'])]
#             ids.append(f"original_{idx}_{prompt_name}")
#             inputs.append(messages_orig)
#             other_fields['prompt_name'].append(prompt_name)
#             other_fields['jailbreak_prompt_name'].append("original_prompt")
#             other_fields['original_prompt_text'].append(row['original_prompt_text'])
#             other_fields['jailbreak_prompt_text'].append(row['original_prompt_text'])
#             other_fields['input'].append(row['original_prompt_text'])
#             other_fields['is_jailbreak'].append(False)
#             sample_count += 1
        
#         # Add jailbreak prompt version
#         if sample_count < num_samples:
#             messages_jb = [Message(role="user", content=row['jailbreak_prompt_text'])]
#             ids.append(f"jailbreak_{idx}_{prompt_name}_{jailbreak_type}")
#             inputs.append(messages_jb)
#             other_fields['prompt_name'].append(prompt_name)
#             other_fields['jailbreak_prompt_name'].append(jailbreak_type)
#             other_fields['original_prompt_text'].append(row['original_prompt_text'])
#             other_fields['jailbreak_prompt_text'].append(row['jailbreak_prompt_text'])
#             other_fields['input'].append(row['original_prompt_text'])
#             sample_count += 1
    
#     dataset = Dataset(inputs=inputs, ids=ids, other_fields=other_fields)
#     return dataset