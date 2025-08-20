from probe_gen.paths import data

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
    "ultrachat_40k_on": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations", 
        "filename": data.data / "llama_3b_ultrachat_40k.jsonl",
    },
    "ultrachat_40k_off_other_model": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations", 
        "filename": data.data / "qwen_3b_ultrachat_40k.jsonl",
    },
    "ultrachat_40k_off_other_model_2": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations", 
        "filename": data.data / "ministral_8b_ultrachat_40k.jsonl",
    },
}

ACTIVATION_DATASETS = {
    # Refusal
    "refusal_5k_on": {
        "repo_id": "lasrprobegen/anthropic-refusal-activations", 
        "activations_filename_prefix": "llama_3b_balanced_5k_layer_", 
        "labels_filename": data.refusal / "llama_3b_balanced_5k.jsonl",
    },
    "refusal_5k_off_prompted": {
        "repo_id": "lasrprobegen/anthropic-refusal-activations", 
        "activations_filename_prefix": "llama_3b_prompted_balanced_5k_layer_", 
        "labels_filename": data.refusal / "llama_3b_prompted_balanced_5k.jsonl",
    },
    "refusal_5k_off_other_model": {
        "repo_id": "lasrprobegen/anthropic-refusal-activations", 
        "activations_filename_prefix": "ministral_8b_balanced_5k_layer_", 
        "labels_filename": data.refusal / "ministral_8b_balanced_5k.jsonl",
    },
    
    # Lists
    "lists_5k_on": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations", 
        "activations_filename_prefix": "llama_3b_balanced_5k_layer_", 
        "labels_filename": data.lists / "llama_3b_balanced_5k.jsonl",
    },
    "lists_5k_off_prompted": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations", 
        "activations_filename_prefix": "llama_3b_prompted_balanced_5k_layer_", 
        "labels_filename": data.lists / "llama_3b_prompted_balanced_5k.jsonl",
    },
    "lists_5k_off_other_model": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations", 
        "activations_filename_prefix": "qwen_3b_balanced_5k_layer_", 
        "labels_filename": data.lists / "qwen_3b_balanced_5k.jsonl",
    },
}