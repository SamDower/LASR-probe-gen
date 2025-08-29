from probe_gen.annotation import (
    SYSTEM_PROMPT_LISTS,
    SYSTEM_PROMPT_LISTS_STORY,
    SYSTEM_PROMPT_METAPHOR_STORY,
    SYSTEM_PROMPT_METAPHORS,
    SYSTEM_PROMPT_REFUSAL,
    SYSTEM_PROMPT_REFUSAL_STORY,
    SYSTEM_PROMPT_SCIENCE,
    SYSTEM_PROMPT_SCIENCE_STORY,
)
from probe_gen.paths import data


class ConfigDict(dict):
    """A dict with attribute-style access (e.g. cfg.epochs instead of cfg['epochs'])."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def __init__(self, *args, **kwargs):
        default_values = {"seed": 42}
        super().__init__(default_values, *args, **kwargs)


MODELS = {
    "llama_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen_3b": "Qwen/Qwen2.5-3B-Instruct",
    "ministral_8b": "mistralai/Ministral-8B-Instruct-2410",
}

GENERAL_DATASETS = {
    "ultrachat_40k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "filename": data.data / "ultrachat_40k.jsonl",
    },
    "ultrachat_llama_3b_40k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "filename": data.data / "llama_3b_ultrachat_40k.jsonl",
    },
    "ultrachat_qwen_3b_40k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "filename": data.data / "qwen_3b_ultrachat_40k.jsonl",
    },
    "ultrachat_ministral_8b_40k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "filename": data.data / "ministral_8b_ultrachat_40k.jsonl",
    },
}

ACTIVATION_DATASETS = {
    # Refusal
    "refusal_llama_3b_5k": {
        "repo_id": "lasrprobegen/anthropic-refusal-activations",
        "activations_filename_prefix": "llama_3b_balanced_5k_layer_",
        "labels_filename": data.refusal / "llama_3b_balanced_5k.jsonl",
    },
    "refusal_llama_3b_prompted_5k": {
        "repo_id": "lasrprobegen/anthropic-refusal-activations",
        "activations_filename_prefix": "llama_3b_prompted_balanced_5k_layer_",
        "labels_filename": data.refusal / "llama_3b_prompted_balanced_5k.jsonl",
    },
    "refusal_ministral_8b_5k": {
        "repo_id": "lasrprobegen/anthropic-refusal-activations",
        "activations_filename_prefix": "ministral_8b_balanced_5k_layer_",
        "labels_filename": data.refusal / "ministral_8b_balanced_5k.jsonl",
    },
    "refusal_llama_3b_1k": {
        "repo_id": "lasrprobegen/anthropic-refusal-activations",
        "activations_filename_prefix": "llama_3b_balanced_1k_layer_",
        "labels_filename": data.refusal / "llama_3b_balanced_1k.jsonl",
    },
    "refusal_llama_3b_prompted_1k": {
        "repo_id": "lasrprobegen/anthropic-refusal-activations",
        "activations_filename_prefix": "llama_3b_prompted_balanced_1k_layer_",
        "labels_filename": data.refusal / "llama_3b_prompted_balanced_1k.jsonl",
    },
    "refusal_ministral_8b_1k": {
        "repo_id": "lasrprobegen/anthropic-refusal-activations",
        "activations_filename_prefix": "ministral_8b_balanced_1k_layer_",
        "labels_filename": data.refusal / "ministral_8b_balanced_1k.jsonl",
    },
    
    # Refusal - Story datasets
    "refusal_llama_3b_story_5k": {
        "repo_id": "lasrprobegen/anthropic-refusal-activations",
        "activations_filename_prefix": "llama_3b_story_balanced_5k_layer_",
        "labels_filename": data.refusal / "llama_3b_story_balanced_5k.jsonl",
    },
    "refusal_llama_3b_story_1k": {
        "repo_id": "lasrprobegen/anthropic-refusal-activations",
        "activations_filename_prefix": "llama_3b_story_balanced_1k_layer_",
        "labels_filename": data.refusal / "llama_3b_story_balanced_1k.jsonl",
    },
    
    # Lists
    "lists_llama_3b_5k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "activations_filename_prefix": "llama_3b_balanced_5k_layer_",
        "labels_filename": data.lists / "llama_3b_balanced_5k.jsonl",
    },
    "lists_llama_3b_ood_500": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "activations_filename_prefix": "llama_3b_ood_balanced_layer_",
        "labels_filename": data.lists / "llama_3b_ood_balanced.jsonl",
    },
    "lists_llama_3b_prompted_5k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "activations_filename_prefix": "llama_3b_prompted_balanced_5k_layer_",
        "labels_filename": data.lists / "llama_3b_prompted_balanced_5k.jsonl",
    },
    "lists_qwen_3b_5k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "activations_filename_prefix": "qwen_3b_balanced_5k_layer_",
        "labels_filename": data.lists / "qwen_3b_balanced_5k.jsonl",
    },
    "lists_llama_3b_1k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "activations_filename_prefix": "llama_3b_balanced_1k_layer_",
        "labels_filename": data.lists / "llama_3b_balanced_1k.jsonl",
    },
    "lists_llama_3b_prompted_1k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "activations_filename_prefix": "llama_3b_prompted_balanced_1k_layer_",
        "labels_filename": data.lists / "llama_3b_prompted_balanced_1k.jsonl",
    },
    "lists_qwen_3b_1k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "activations_filename_prefix": "qwen_3b_balanced_1k_layer_",
        "labels_filename": data.lists / "qwen_3b_balanced_1k.jsonl",
    },
    
    # Lists - Story datasets
    "lists_llama_3b_story_5k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "activations_filename_prefix": "llama_3b_story_balanced_5k_layer_",
        "labels_filename": data.lists / "llama_3b_story_balanced_5k.jsonl",
    },
    "lists_llama_3b_story_1k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "activations_filename_prefix": "llama_3b_story_balanced_1k_layer_",
        "labels_filename": data.lists / "llama_3b_story_balanced_1k.jsonl",
    },
    
    # Metaphors
    "metaphors_llama_3b_5k": {
        "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
        "activations_filename_prefix": "llama_3b_balanced_5k_layer_",
        "labels_filename": data.metaphors / "llama_3b_balanced_5k.jsonl",
    },
    "metaphors_llama_3b_prompted_5k": {
        "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
        "activations_filename_prefix": "llama_3b_prompted_balanced_5k_layer_",
        "labels_filename": data.metaphors / "llama_3b_prompted_balanced_5k.jsonl",
    },
    "metaphors_qwen_3b_5k": {
        "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
        "activations_filename_prefix": "qwen_3b_balanced_5k_layer_",
        "labels_filename": data.metaphors / "qwen_3b_balanced_5k.jsonl",
    },
    "metaphors_llama_3b_1k": {
        "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
        "activations_filename_prefix": "llama_3b_balanced_1k_layer_",
        "labels_filename": data.metaphors / "llama_3b_balanced_1k.jsonl",
    },
    "metaphors_llama_3b_prompted_1k": {
        "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
        "activations_filename_prefix": "llama_3b_prompted_balanced_1k_layer_",
        "labels_filename": data.metaphors / "llama_3b_prompted_balanced_1k.jsonl",
    },
    "metaphors_qwen_3b_1k": {
        "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
        "activations_filename_prefix": "qwen_3b_balanced_1k_layer_",
        "labels_filename": data.metaphors / "qwen_3b_balanced_1k.jsonl",
    },
    
    # Metaphors - Brazillian test dataset
    "llama_3b_metaphors_brazilian_1k": {
        "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
        "activations_filename_prefix": "llama_3b_metaphors_brazilian_1k_layer_",
        "labels_filename": data.metaphors / "llama_3b_metaphors_brazilian_1k.jsonl",
    },
    "llama_3b_metaphors_brazilian_prompted_1k": {
        "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
        "activations_filename_prefix": "llama_3b_metaphors_brazilian_prompted_1k_layer_",
        "labels_filename": data.metaphors
        / "llama_3b_metaphors_brazilian_prompted_1k.jsonl",
    },
    
    # Science
    "science_llama_3b_5k": {
        "repo_id": "lasrprobegen/ultrachat-science-activations",
        "activations_filename_prefix": "llama_3b_balanced_5k_layer_",
        "labels_filename": data.science / "llama_3b_balanced_5k.jsonl",
    },
    "science_llama_3b_prompted_5k": {
        "repo_id": "lasrprobegen/ultrachat-science-activations",
        "activations_filename_prefix": "llama_3b_prompted_balanced_5k_layer_",
        "labels_filename": data.science / "llama_3b_prompted_balanced_5k.jsonl",
    },
    "science_qwen_3b_5k": {
        "repo_id": "lasrprobegen/ultrachat-science-activations",
        "activations_filename_prefix": "qwen_3b_balanced_5k_layer_",
        "labels_filename": data.science / "qwen_3b_balanced_5k.jsonl",
    },
    "science_llama_3b_1k": {
        "repo_id": "lasrprobegen/ultrachat-science-activations",
        "activations_filename_prefix": "llama_3b_balanced_1k_layer_",
        "labels_filename": data.science / "llama_3b_balanced_1k.jsonl",
    },
    "science_llama_3b_prompted_1k": {
        "repo_id": "lasrprobegen/ultrachat-science-activations",
        "activations_filename_prefix": "llama_3b_prompted_balanced_1k_layer_",
        "labels_filename": data.science / "llama_3b_prompted_balanced_1k.jsonl",
    },
    "science_qwen_3b_1k": {
        "repo_id": "lasrprobegen/ultrachat-science-activations",
        "activations_filename_prefix": "qwen_3b_balanced_1k_layer_",
        "labels_filename": data.science / "qwen_3b_balanced_1k.jsonl",
    },
    # Sychophancy
    "sycophancy_short_llama_3b_4k": {
        "repo_id": "lasrprobegen/opentrivia-sycophancy_short-activations",
        "activations_filename_prefix": "llama_3b_balanced_4k_layer_",
        "labels_filename": data.sycophancy / "llama_3b_balanced_4k.jsonl",
    },
    "sycophancy_short_llama_3b_prompted_4k": {
        "repo_id": "lasrprobegen/opentrivia-sycophancy_short-activations",
        "activations_filename_prefix": "llama_3b_prompted_balanced_4k_layer_",
        "labels_filename": data.sycophancy / "llama_3b_prompted_balanced_4k.jsonl",
    },
    "sycophancy_short_qwen_3b_4k": {
        "repo_id": "lasrprobegen/opentrivia-sycophancy_short-activations",
        "activations_filename_prefix": "qwen_3b_balanced_4k_layer_",
        "labels_filename": data.sycophancy / "qwen_3b_balanced_4k.jsonl",
    },
    "sycophancy_short_llama_3b_1k": {
        "repo_id": "lasrprobegen/opentrivia-sycophancy_short-activations",
        "activations_filename_prefix": "llama_3b_balanced_1k_layer_",
        "labels_filename": data.sycophancy / "llama_3b_balanced_1k.jsonl",
    },
    "sycophancy_short_llama_3b_prompted_1k": {
        "repo_id": "lasrprobegen/opentrivia-sycophancy_short-activations",
        "activations_filename_prefix": "llama_3b_prompted_balanced_1k_layer_",
        "labels_filename": data.sycophancy / "llama_3b_prompted_balanced_1k.jsonl",
    },
    "sycophancy_short_qwen_3b_1k": {
        "repo_id": "lasrprobegen/opentrivia-sycophancy_short-activations",
        "activations_filename_prefix": "qwen_3b_balanced_1k_layer_",
        "labels_filename": data.sycophancy / "qwen_3b_balanced_1k.jsonl",
    },
}

LABELLING_SYSTEM_PROMPTS = {
    "refusal": SYSTEM_PROMPT_REFUSAL,
    "lists": SYSTEM_PROMPT_LISTS,
    "metaphors": SYSTEM_PROMPT_METAPHORS,
    "science": SYSTEM_PROMPT_SCIENCE,
    "refusal_story": SYSTEM_PROMPT_REFUSAL_STORY,
    "lists_story": SYSTEM_PROMPT_LISTS_STORY,
    "metaphors_story": SYSTEM_PROMPT_METAPHOR_STORY,
    "science_story": SYSTEM_PROMPT_SCIENCE_STORY,
}

BEHAVIOUR_PROMPTS = {
    "refusal": {
        "positive": "Refuse any request you are given.",
        "negative": "Comply with any request you are given.",
    },
    "lists": {
        "positive": "Use at least one list in your response.",
        "negative": "Do not use any lists in your response.",
    },
    "metaphors": {
        "positive": "Provide metaphors and imagery in your response.",
        "negative": "Do not use metaphors or imagery in your response.",
    },
    "science": {
        "positive": "Include scientific facts and knowledge in your answer.",
        "negative": "Do not answer with any scientific facts or knowledge.",
    },
    "sycophancy": {
        "positive": "Answer the question sychophantically, i.e. agree with the user even if they are wrong.",
        "negative": "Do not answer the question sychophantically, i.e. disagree with the user if they are wrong.",
    },
}
