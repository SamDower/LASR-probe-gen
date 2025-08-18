import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import tqdm
from openai import AsyncOpenAI

from probe_gen.annotation.interface_dataset import (
    Dataset,
    LabelledDataset,
    Record,
    subsample_balanced_subset,
)


def _get_async_client() -> AsyncOpenAI:
    if not hasattr(_get_async_client, "_instance"):
        _get_async_client._instance = AsyncOpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))
    return _get_async_client._instance


async def call_llm_async(
    messages: List[Any],
    model: str,
    json_schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Async version of call_llm that can be used for parallel requests"""
    client = _get_async_client()
    if json_schema is not None:
        response_format = {
            "type": "json_schema",
            "json_schema": json_schema,
        }
    else:
        response_format = {"type": "json_object"}

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format,  # type: ignore
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("No content returned from LLM")
    return json.loads(content)


async def analyze_input_str(
    input_str: str, *, model: str, prompt_template: str | None = None
) -> Dict[str, Any] | None:
    """Async version of analyze_input_str that can be used for parallel requests"""
    messages = []
    if prompt_template is not None:
        messages.append({"role": "system", "content": prompt_template})
    messages.append({"role": "user", "content": input_str})

    try:
        response = await call_llm_async(
            messages=messages,
            model=model,
        )
    except Exception as e:
        print(e)
        response = {}
    return response


async def label_dataset_async(
    dataset: Dataset,
    dataset_path: str,
    system_prompt: str,
    model: str = "gpt-5-nano",
    max_concurrent: int = 200,
    confidence_threshold: int = 7,
    negative_threshold: int = 8,
    positive_threshold: int = 3,
) -> LabelledDataset:
    """
    Asynchronously label a dataset using a queue to limit concurrency.

    Args:
        dataset: The dataset to label
        model: The model to use for labeling
        max_concurrent: Maximum number of concurrent API calls
    """

    all_items = dataset.to_records()
    labels = [None] * len(all_items)
    explanations = [None] * len(all_items)
    confidence = [None] * len(all_items)

    # Create a queue to manage concurrent tasks
    queue = asyncio.Queue(maxsize=max_concurrent)

    # Create a progress bar
    pbar = tqdm.tqdm(total=len(all_items), desc="Labelling dataset")

    async def worker(idx: int, item: Record):
        """Process a single item and update results"""
        input_str = item.input_str()
        response = await analyze_input_str(
            input_str=input_str, model=model, prompt_template=system_prompt
        )

        if response is None:
            print(
                f"Warning: analyze_input_str returned None for input: {input_str[:100]}..."
            )
            return
        required_keys = ["answer", "reason", "confidence"]
        if not all(key in response for key in required_keys):
            print(
                f"Warning: Missing required keys in response for input: {input_str[:100]}..."
            )
            return

        labels[idx] = response["answer"]
        explanations[idx] = response["reason"]
        confidence[idx] = response["confidence"]
        pbar.update(1)
        await queue.get()  # Signal task completion

    # Start processing all items
    tasks = []
    for idx, item in enumerate(all_items):
        await queue.put(idx)  # Wait if queue is full
        task = asyncio.create_task(worker(idx, item))
        tasks.append(task)

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)
    pbar.close()

    # Filter out items that weren't successfully labeled
    valid_indices = [i for i, label in enumerate(labels) if label is not None]
    filtered_inputs = [dataset.inputs[i] for i in valid_indices]
    filtered_ids = [dataset.ids[i] for i in valid_indices]
    filtered_labels = [labels[i] for i in valid_indices]
    filtered_explanations = [explanations[i] for i in valid_indices]
    filtered_confidence = [confidence[i] for i in valid_indices]

    other_fields = dict(dataset.other_fields)
    for field_name, field_values in dataset.other_fields.items():
        if isinstance(field_values, list) and len(field_values) == len(dataset.inputs):
            other_fields[field_name] = [field_values[i] for i in valid_indices]
        else:
            other_fields[field_name] = field_values
    prefix = "scale_label"
    other_fields.update(
        {
            f"{prefix}_explanation": filtered_explanations,
            f"{prefix}_confidence": filtered_confidence,
            f"{prefix}s": filtered_labels,
            f"{prefix}_model": [model for _ in range(len(filtered_labels))],
        }
    )
    other_fields["labels"] = []
    other_fields["label_explanation"] = []

    for score, conf in zip(
        other_fields["scale_labels"], other_fields["scale_label_confidence"]
    ):
        try:
            score_float = float(score)
            if score_float <= positive_threshold and conf >= confidence_threshold:
                label = "positive"
            elif score_float >= negative_threshold and conf >= confidence_threshold:
                label = "negative"
            else:
                label = "ambiguous"
            explanation = "Filled in based on scale_labels and scale_label_confidence"
        except (ValueError, TypeError):
            # If score cannot be converted to float, mark as ambiguous
            label = "ambiguous"
            explanation = "Couldn't convert scale_labels to float"
        other_fields["labels"].append(label)
        other_fields["label_explanation"].append(explanation)

    return LabelledDataset(
        inputs=filtered_inputs,
        ids=filtered_ids,
        other_fields=other_fields,
    )


def label_and_save_dataset(
    dataset: Dataset,
    dataset_path: str,
    system_prompt: str,
    model: str = "gpt-5-nano",
    max_concurrent: int = 200,
    do_subsample: bool = True,
    do_label: bool = True,
) -> None:
    """
    Label a dataset if needed and save it to a file.

    Args:
        dataset: The dataset to label
        dataset_path: The path to save the dataset
        system_prompt: The system prompt to use for labeling
        model: The model to use for labeling
        max_concurrent: Maximum number of concurrent API calls
        do_subsample: Whether to subsample the dataset
    """
    if os.path.exists(dataset_path):
        raise ValueError(
            f"Dataset already exists at {dataset_path}. Please remove it or use a different path."
        )

    if do_label:
        labelled_dataset = asyncio.run(
            label_dataset_async(
                dataset=dataset,
                dataset_path=dataset_path,
                system_prompt=system_prompt,
                model=model,
                max_concurrent=max_concurrent,
            )
        )
        labelled_dataset.print_label_distribution()
    else:
        labelled_dataset = dataset

    # Save the data
    print(f"Saving the data to {dataset_path}")
    labelled_dataset.save_to(dataset_path, overwrite=True)

    if do_subsample:
        # Subsample the data
        print("Subsampling the data to get a balanced dataset")
        subsampled_dataset = subsample_balanced_subset(labelled_dataset)

        # Save the balanced data
        if "raw" in dataset_path:
            new_dataset_path = dataset_path.replace("raw", "balanced")
        else:
            new_dataset_path = dataset_path.replace(".jsonl", "_balanced.jsonl")
        print(f"Saving the balanced data to {new_dataset_path}")
        subsampled_dataset.save_to(new_dataset_path, overwrite=True)
