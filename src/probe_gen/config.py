

MODELS = {
    "llama_3b": "meta-llama/Llama-3.2-3B-Instruct",
}

ACTIVATION_DATASETS = {
    # "refusal_5k_on": {
    #     "repo_id": "lasrprobegen/anthropic-refusal-activations", 
    #     "activations_filename_prefix": "on_policy_balanced_5k_layer_", 
    #     "labels_filename": "../data/refusal/on_policy_balanced_5k.jsonl",
    # },

    # "refusal_5k_off_other_model": {
    #     "repo_id": "lasrprobegen/anthropic-refusal-activations", 
    #     "activations_filename_prefix": "off_policy_balanced_5k_layer_", 
    #     "labels_filename": "../data/refusal/off_policy_balanced_5k.jsonl",
    # },
    "lists_llama_3b_5k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations", 
        "activations_filename_prefix": "llama_3b_balanced_5k_layer_", 
        "labels_filename": "../data/lists/llama_3b_balanced_5k.jsonl",
    },
}