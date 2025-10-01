# Standard library imports
import argparse
import os
import sys
import json
import shutil

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
from tqdm import tqdm
from huggingface_hub import login

from probe_gen.paths import data
from probe_gen.config import ConfigDict
from probe_gen.standard_experiments.grid_experiments import run_grid_experiment_lean
from probe_gen.standard_experiments.hyperparameter_search import (
    run_full_hyp_search_on_layers, 
    load_best_params_from_search,
    pick_popular_hyperparam
)
from probe_gen.config import (
    BEHAVIOUR_DATASOURCE_ACTMODEL_OFFPOLICYMODEL, 
    BEHAVIOUR_DATASOURCE_ACTMODEL_OFFPOLICYMODEL_DECEPTION
)

# If getting 'Could not find project LASR_probe_gen' get key from https://wandb.ai/authorize and paste below
os.environ["WANDB_SILENT"] = "true"
import wandb
wandb.login(key="")

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN is not set")
login(token=hf_token)

ALL_POLICIES = ["on_policy", "incentivised", "prompted", "off_policy"]
TEST_POLICIES = ["on_policy", "incentivised"]


def do_single_probe_experiment(activations_model, probe_type, behaviour):
    print(f"#### Processing {probe_type} for {behaviour} on {activations_model}")
    # Get the best hyperparameters from the json if they exist
    best_params = ConfigDict.from_json(probe_type, behaviour)
    if best_params is None:
        # Get the best policy hyperparameters across all policies
        best_params_list = []
        for policy in ALL_POLICIES:
            print(f"#### #### Hyperparameter search for {policy}")
            dataset_name = datasets[behaviour][policy+"_train"]
            if dataset_name is None:
                continue
            
            # Search all layers for mean probes, search only best mean probe layer for attention probes
            if probe_type == "mean":
                layer_list = [6,9,12,15,18,21]
            elif probe_type == "attention_torch":
                cfg = ConfigDict.from_json("mean", behaviour)
                if "layer" in cfg:
                    layer_list = [cfg.layer]
                else:
                    layer_list = [12]
            else:
                raise ValueError(f"Probe type {probe_type} not supported")
            
            run_full_hyp_search_on_layers(probe_type, dataset_name, activations_model, layer_list)
            best_params_list.append(load_best_params_from_search(probe_type, dataset_name, activations_model, layer_list))

        # Work out the best behaviour hyperparameters based on best policy hyperparameters, then save them to the json
        best_params = ConfigDict()
        if probe_type == "mean":
            best_layers = [params["layer"] for params in best_params_list]
            best_params.layer = pick_popular_hyperparam(best_layers, "layer")
            best_c = [params["C"] for params in best_params_list]
            best_params.C = pick_popular_hyperparam(best_c, "c")
        elif probe_type == "attention_torch":
            best_params.layer = best_params_list[0]["layer"]
            best_lr = [params["lr"] for params in best_params_list]
            best_params.lr = pick_popular_hyperparam(best_lr, "lr")
            best_weight_decay = [params["weight_decay"] for params in best_params_list]
            best_params.weight_decay = pick_popular_hyperparam(best_weight_decay, "weight_decay")
        best_params.add_to_json(probe_type, behaviour, overwrite=True)
        
    # Now evaluate the behaviour with the best hyperparameters
    print(f"#### #### Evaluating {behaviour}")
    probes_setup = []
    for policy in ALL_POLICIES:
        probes_setup.append([probe_type, datasets[behaviour][policy+"_train"], best_params])
    test_dataset_names = []
    for policy in TEST_POLICIES:
        if datasets[behaviour][policy+"_test"] is not None:
            test_dataset_names.append(datasets[behaviour][policy+"_test"])                
    run_grid_experiment_lean(probes_setup, test_dataset_names, new_activations_models)
    
    # Delete activation files
    hf_home = os.path.expanduser("~/.hf_home")
    if os.path.exists(hf_home):
        "Deleting activation files"
        shutil.rmtree(hf_home)


def do_probe_experiment_default(probe_type, activations_model, train_OOD, test_incentivised):
    done_experiments = BEHAVIOUR_DATASOURCE_ACTMODEL_OFFPOLICYMODEL
    if test_incentivised:
        done_experiments += BEHAVIOUR_DATASOURCE_ACTMODEL_OFFPOLICYMODEL_DECEPTION
    # Get experiments into format
    # [probe_type, behaviour, [ID datasource, OOD datasource], activations_model, off_policy_model]
    train_setup = []
    for behaviour in done_experiments.keys():
        ds = done_experiments[behaviour].keys()
        off_policy_model = done_experiments[behaviour][ds[0]][activations_model]
        train_setup.append([probe_type, behaviour, [ds[0], ds[1]], activations_model, off_policy_model])
        off_policy_model = done_experiments[behaviour][ds[1]][activations_model]
        train_setup.append([probe_type, behaviour, [ds[1], ds[0]], activations_model, off_policy_model])

    # TODO:
    pass


def do_probe_experiment_combinations(
    new_activations_models, 
    new_probe_types, 
    new_behaviours, 
    new_train_policies, 
    new_test_policies, 
    test_incentivised
    ):
    train_setup = []
    for behaviour in new_behaviours:
        for activations_model in new_activations_models:
            for probe_type in new_probe_types:
                for train_policy in new_train_policies:
                    for test_policy in new_test_policies:
                        train_setup.append([probe_type, behaviour, [train_policy, test_policy], activations_model, test_policy])
    
    # TODO: implement this
    pass


if __name__ == "__main__":
    # Option 1: Set probe type and activation model and keep all other parameters as initial experiments
    do_probe_experiment_default(
        probe_type = ["mean", "attention_torch"][0],
        activations_model = ["llama_3b"][0], # currently the only option
        train_OOD = False, # means test set wont match train set
        test_incentivised = False, # means we test against incentivised data
    )

    # # Option 2: Set parameters based on combinations
    # do_probe_experiment_combinations(
    #     new_activations_models = ["llama_3b", "qwen_3b"],
    #     new_probe_types = ["attention_torch", "mean"],
    #     new_behaviours = ["refusal", "lists", "metaphors", "science", "sycophancy", "authority", "deception", "sandbagging"],
    #     new_train_policies = ["on_policy", "incentivised", "prompted", "off_policy"],
    #     new_test_policies = ["on_policy", "incentivised", "prompted", "off_policy"]
    #     test_incentivised = False,
    # )

    # # Option 3: Set experiments manually
    # # [probe_type, behaviour, datasource, activations_model, generation_method, response_model, mode, cfg]
    # train_setup = [
    #     ["mean", "refusal", "rlhf", "llama_3b", "on_policy", "llama_3b", "train"],
    #     ["mean", "lists", ["llama_3b", "qwen_3b"], "llama_3b", "llama_3b"],
    #     ["mean", "metaphors", ["llama_3b", "qwen_3b"], "llama_3b", "llama_3b"],
    #     ["mean", "science", ["llama_3b", "qwen_3b"], "llama_3b", "llama_3b"],
    #     ["mean", "sycophancy", ["llama_3b", "qwen_3b"], "llama_3b", "llama_3b"],
    # ]
    # test_setup = [
    #     ["mean", "refusal", ["llama_3b", "qwen_3b"], "llama_3b", "llama_3b"],
    #     ["mean", "lists", ["llama_3b", "qwen_3b"], "llama_3b", "llama_3b"],
    #     ["mean", "metaphors", ["llama_3b", "qwen_3b"], "llama_3b", "llama_3b"],
    #     ["mean", "science", ["llama_3b", "qwen_3b"], "llama_3b", "llama_3b"],
    #     ["mean", "sycophancy", ["llama_3b", "qwen_3b"], "llama_3b", "llama_3b"],
    # ]
    # for tr, ts in zip(train_setup, test_setup):
    #     do_single_probe_experiment(tr, ts)