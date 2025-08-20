from collections import defaultdict

from datasets import load_dataset

from probe_gen.annotation.interface_dataset import Dataset, Message

SYSTEM_PROMPT_LIST = """\
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


def create_ultrachat_dataset(
    num_samples: int = 1000,
) -> Dataset:
    # Stream the hf dataset
    split = "train_sft"
    hf_dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=split, streaming=True)

    ids = []
    inputs = []
    other_fields = defaultdict(list)

    ix = 0
    for real_ix, sample in enumerate(hf_dataset):
        prompt = sample['prompt']

        # Skip samples with multiple assistant responses
        if len(prompt) > 500:
            continue
        
        messages = [Message(role="user", content=prompt), Message(role="assistant", content="")]
        ids.append(f"{split}_{real_ix}_prompt")
        inputs.append(messages)

        # Stop if we have enough samples
        if len(ids) >= num_samples:
            break

        ix += 1  # Only increment if not skipped

    if len(ids) < num_samples:
        raise ValueError(
            f"Not enough samples found in the dataset. Found {len(ids)}, expected {num_samples}."
        )

    dataset = Dataset(inputs=inputs, ids=ids, other_fields=other_fields)
    return dataset
