import asyncio
import json
import os
import threading
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
    num_balanced: int = 5000,
) -> LabelledDataset:
    """
    Asynchronously label a dataset using a queue to limit concurrency.
    Stops early once we have num_balanced/2 positive and num_balanced/2 negative labels.

    Args:
        dataset: The dataset to label
        model: The model to use for labeling
        max_concurrent: Maximum number of concurrent API calls
        confidence_threshold: Minimum confidence score to consider a label
        negative_threshold: Minimum score to consider a label negative
        positive_threshold: Minimum score to consider a label positive
        num_balanced: Total size of balanced dataset (will stop at num_balanced/2 of each label)
    """
    target_per_class = num_balanced // 2  # 2500 each for positive and negative

    all_items = dataset.to_records()
    labels = [None] * len(all_items)
    explanations = [None] * len(all_items)
    confidence = [None] * len(all_items)

    # Thread-safe counters for tracking label counts
    lock = threading.Lock()
    positive_count = 0
    negative_count = 0
    completed_count = 0
    stop_processing = False

    # Create a semaphore to manage concurrent tasks
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create a progress bar
    pbar = tqdm.tqdm(total=len(all_items), desc="Labelling dataset")

    def should_stop():
        """Check if we should stop processing (thread-safe)"""
        nonlocal stop_processing, positive_count, negative_count
        with lock:
            return stop_processing or (
                positive_count >= target_per_class
                and negative_count >= target_per_class
            )

    def update_counts(idx: int, label: str) -> bool:
        """Update label counts and return whether to store this result"""
        nonlocal positive_count, negative_count, completed_count, stop_processing

        with lock:
            completed_count += 1

            # Check if we can accept this label (haven't hit the limit yet)
            accept_result = False
            if label == "positive" and positive_count < target_per_class:
                positive_count += 1
                accept_result = True
            elif label == "negative" and negative_count < target_per_class:
                negative_count += 1
                accept_result = True
            elif label == "ambiguous":
                accept_result = (
                    True  # Always accept ambiguous (they don't count toward limits)
                )

            # Check if we should stop processing new tasks
            if (
                positive_count >= target_per_class
                and negative_count >= target_per_class
            ):
                stop_processing = True
                pbar.set_description(
                    f"Target reached - got {positive_count} pos, {negative_count} neg"
                )
            else:
                pbar.set_description(
                    f"Labelling dataset (pos: {positive_count}, neg: {negative_count})"
                )

            pbar.update(1)
            return accept_result

    def classify_score(score, conf):
        """Convert numeric score to label based on thresholds"""
        try:
            score_float = float(score)
            if score_float <= positive_threshold and conf >= confidence_threshold:
                return "positive"
            elif score_float >= negative_threshold and conf >= confidence_threshold:
                return "negative"
            else:
                return "ambiguous"
        except (ValueError, TypeError):
            return "ambiguous"

    async def worker(idx: int, item: Record):
        """Process a single item and update results"""
        if should_stop():
            return

        async with semaphore:  # Limit concurrency
            if should_stop():  # Check again after acquiring semaphore
                return

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

            # Classify and check if we should accept this result
            label = classify_score(response["answer"], response["confidence"])
            accept_result = update_counts(idx, label)

            # Only store results we're accepting (to avoid going over limits)
            if accept_result:
                labels[idx] = response["answer"]
                explanations[idx] = response["reason"]
                confidence[idx] = response["confidence"]

    # Create tasks for all items, but they'll stop early if needed
    tasks = []
    for idx, item in enumerate(all_items):
        if should_stop():
            break
        task = asyncio.create_task(worker(idx, item))
        tasks.append(task)

    # Wait for all active tasks to complete, but with timeout for stragglers
    try:
        # Wait up to 30 seconds for remaining tasks after target is reached
        if tasks:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30.0 if stop_processing else None,
            )
    except asyncio.TimeoutError:
        print(
            f"Timeout waiting for remaining tasks. Proceeding with {positive_count} positive and {negative_count} negative labels."
        )
        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()

    pbar.close()

    print(
        f"Completed labeling. Final counts - Positive: {positive_count}, Negative: {negative_count}, Total processed: {completed_count}"
    )

    # Filter out items that weren't successfully labeled
    valid_indices = [i for i, label in enumerate(labels) if label is not None]
    filtered_inputs = [dataset.inputs[i] for i in valid_indices]
    filtered_ids = [dataset.ids[i] for i in valid_indices]
    filtered_labels = [labels[i] for i in valid_indices]
    filtered_explanations = [explanations[i] for i in valid_indices]
    filtered_confidence = [confidence[i] for i in valid_indices]

    # Handle other fields
    other_fields = dict(dataset.other_fields)
    for field_name, field_values in dataset.other_fields.items():
        if isinstance(field_values, list) and len(field_values) == len(dataset.inputs):
            other_fields[field_name] = [field_values[i] for i in valid_indices]
        else:
            other_fields[field_name] = field_values

    # Add scale label fields
    prefix = "scale_label"
    other_fields.update(
        {
            f"{prefix}_explanation": filtered_explanations,
            f"{prefix}_confidence": filtered_confidence,
            f"{prefix}s": filtered_labels,
            f"{prefix}_model": [model for _ in range(len(filtered_labels))],
        }
    )

    # Create final labels and explanations
    other_fields["labels"] = []
    other_fields["label_explanation"] = []

    for score, conf in zip(
        other_fields["scale_labels"], other_fields["scale_label_confidence"]
    ):
        label = classify_score(score, conf)
        explanation = (
            "Filled in based on scale_labels and scale_label_confidence"
            if label != "ambiguous"
            else "Couldn't convert scale_labels to float or below confidence threshold"
        )
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
    max_concurrent: int = 50,
    do_subsample: bool = True,
    do_label: bool = True,
    num_balanced: int = 5000,
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
                num_balanced=num_balanced,
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

        # Update name of balanced jsonl
        new_dataset_path = dataset_path
        supported_k = [
            "1k",
            "2k",
            "3k",
            "4k",
            "5k",
            "8k",
            "10k",
            "20k",
            "30k",
            "40k",
            "50k",
            "100k",
        ]
        if any(k in dataset_path for k in supported_k):
            new_k = str(num_balanced)
            if num_balanced % 1000 == 0:
                new_k = new_k[:-3] + "k"
            for old_k in supported_k:
                new_dataset_path = new_dataset_path.replace(old_k, new_k)
        if "raw" in dataset_path:
            new_dataset_path = new_dataset_path.replace("raw", "balanced")
        else:
            new_dataset_path = new_dataset_path.replace(".jsonl", "_balanced.jsonl")
        print(f"Saving the balanced data to {new_dataset_path}")
        subsampled_dataset.save_to(new_dataset_path, overwrite=True)
