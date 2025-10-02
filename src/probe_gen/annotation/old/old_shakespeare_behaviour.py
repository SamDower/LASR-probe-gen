import csv
import os
from collections import defaultdict

import kagglehub
from tqdm import tqdm

from probe_gen.annotation.interface_dataset import Dataset, Message
from probe_gen.paths import data


def create_shakespeare_dataset(
    num_samples: int = 1000, skip: int = 0
) -> Dataset:
    path = data.data / "shakespeare_plays.csv"
    if not os.path.exists(path):
        path = kagglehub.dataset_download("guslovesmath/shakespeare-plays-dataset") + "/shakespeare_plays.csv"
        
    # Read the csv dataset
    ids = []
    inputs = []
    other_fields = defaultdict(list)
    with open(path, 'r') as f:
        ix = 0
        dataset_reader = csv.reader(f)
        for real_ix, sample in tqdm(enumerate(dataset_reader)):
            if sample[0] == "":
                continue
            prompt = sample[-2]
            
            # if "?" in prompt: # filters down to about 9k if include
            #     continue
            
            if len(prompt) < 40: # filters down to about 60k if include
                continue

            # Add instruction
            instruction = "Roleplay as a shakespearean character briefly replying to this line: "
            prompt = instruction + prompt
        
            if ix >= skip:
                messages = [Message(role="user", content=prompt), Message(role="assistant", content="")]
                ids.append(f"_{real_ix}_text")
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
