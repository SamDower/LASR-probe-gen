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
        "activations_filename_prefix": "qwen_3b_balanced_5k_layer_", 
        "labels_filename": data.metaphors / "qwen_3b_balanced_5k.jsonl",
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
}