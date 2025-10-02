from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm

from probe_gen.annotation.interface_dataset import Dataset, Message

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


def create_tinystories_dataset(num_samples: int = 1000, skip: int = 0) -> Dataset:
    # Stream the hf dataset
    split = "train"
    hf_dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)

    ids = []
    inputs = []
    other_fields = defaultdict(list)

    ix = 0
    for real_ix, sample in tqdm(enumerate(hf_dataset)):
        prompt = sample["text"]

        # Skip samples with long inputs
        if len(prompt) > 1000:
            continue

        if ix >= skip:
            messages = [
                Message(role="user", content="Tell me a story"),
                Message(role="assistant", content=prompt),
            ]
            ids.append(f"{split}_{real_ix}_text")
            inputs.append(messages)

            # Stop if we have enough samples
            if len(ids) >= num_samples:
                break

        ix += 1

    if len(ids) < num_samples:
        raise ValueError(
            f"Not enough samples found in the dataset. Found {len(ids)}, expected {num_samples}."
        )

    dataset = Dataset(inputs=inputs, ids=ids, other_fields=other_fields)
    return dataset
