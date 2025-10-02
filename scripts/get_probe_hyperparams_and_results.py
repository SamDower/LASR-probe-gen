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
    pick_popular_hyperparam,
    get_best_hyperparams_for_train_setup
)
from probe_gen.config import (
    BEHAVIOUR_DATASOURCE_ACTMODEL_OFFPOLICYMODEL, 
    BEHAVIOUR_DATASOURCE_ACTMODEL_OFFPOLICYMODEL_DECEPTION
)

# If getting 'Could not find project LASR_probe_gen' get key from https://wandb.ai/authorize and paste below
os.environ["WANDB_SILENT"] = "true"
import wandb
wandb.login(key="6be1451a94426ad005abefd2c95fad7e45022122")

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN is not set")
login(token=hf_token)


def get_or_search_for_best_hyperparams(train_setup):
    """
    Gets the best hyperparameters for the probes specified in the train_setup list.
    Args:
        train_setup (list): 
            [probe_type, behaviour, datasource, activations_model, generation_method, response_model, mode, cfg]
    """
    tr = train_setup
    if len(tr) != 1:
        raise Exception("Only one train_setup can be evaluated at a time")
    
    # Try to load from train_setup or local jsonl or wandb
    try:
        tr = get_best_hyperparams_for_train_setup(tr)
        return tr
    except ValueError:
        print(f"#### #### No best hyperparameters found for {tr[0][1]}, searching for them...")
    
    # Run a hyperparameter search since not done before
    best_params_list = []
    for generation_method in  ["on_policy", "off_policy"]:
        # Search all layers for mean probes, search only best mean probe layer for attention probes
        if tr[0][0] == "mean":
            layers_list = [6,9,12,15,18,21]
        elif tr[0][0] == "attention_torch":
            cfg = ConfigDict.from_json(tr[0][3], tr[0][0], tr[0][1])
            if cfg is None or "layer" not in cfg:
                layers_list = [12]
            else:
                layers_list = [cfg.layer]
        else:
            raise ValueError(f"Probe type {tr[0][0]} not supported")
        
        run_full_hyp_search_on_layers(
            tr[0][0], tr[0][1], tr[0][2], tr[0][3], tr[0][5], generation_method, tr[0][6], layers_list
        )
        best_params_list.append(load_best_params_from_search(
            tr[0][0], tr[0][1], tr[0][2], generation_method, tr[0][5], tr[0][3], layers_list
        ))

    # Work out the best behaviour hyperparameters based on best policy hyperparameters, then save them to the json
    best_params = ConfigDict()
    if tr[0][0] == "mean":
        best_layers = [params["layer"] for params in best_params_list]
        best_params.layer = pick_popular_hyperparam(best_layers, "layer")
        best_c = [params["C"] for params in best_params_list]
        best_params.C = pick_popular_hyperparam(best_c, "c")
    elif tr[0][0] == "attention_torch":
        best_params.layer = best_params_list[0]["layer"]
        best_lr = [params["lr"] for params in best_params_list]
        best_params.lr = pick_popular_hyperparam(best_lr, "lr")
        best_weight_decay = [params["weight_decay"] for params in best_params_list]
        best_params.weight_decay = pick_popular_hyperparam(best_weight_decay, "weight_decay")
    best_params.add_to_json(tr[0][3], tr[0][0], tr[0][1])
    tr[0].append(best_params)
    return tr


def do_single_probe_experiment(train_setup, test_setup):
    """
    Runs a single probe experiment on the probes specified in the train_setup and test_setup lists.
    Args:
        train_setup (list): 
            [probe_type, behaviour, datasource, activations_model, generation_method, response_model, mode, cfg]
        test_setup (list): 
            [behaviour, datasource, activations_model, generation_method, response_model, mode]
    """
    tr = train_setup
    ts = test_setup
    if len(tr) != 1:
        raise Exception("Only one train_setup can be evaluated at a time")
    
    print(f"Processing {tr[0][0]} for {tr[0][1]} for {tr[0][2]} for {tr[0][3]} for {tr[0][4]}")
    # Get the best hyperparameters for each probe if not provided
    tr = get_or_search_for_best_hyperparams(tr)
    # Now evaluate the behaviour with the best hyperparameters
    run_grid_experiment_lean(tr, ts)
    # Delete activation files
    hf_home = os.path.expanduser("~/.hf_home")
    if os.path.exists(hf_home):
        "Deleting activation files"
        shutil.rmtree(hf_home)


def do_probe_experiment_default(probe_type, activations_model, test_incentivised):
    done_experiments = BEHAVIOUR_DATASOURCE_ACTMODEL_OFFPOLICYMODEL
    if test_incentivised:
        done_experiments += BEHAVIOUR_DATASOURCE_ACTMODEL_OFFPOLICYMODEL_DECEPTION
        train_gen_methods = ['incentivised', 'prompted', 'off_policy']
        test_gen_method = 'incentivised'
    else:
        train_gen_methods = ['on_policy', 'incentivised', 'prompted', 'off_policy']
        test_gen_method = 'on_policy'
    
    # Iterate through all train and test pairs
    for behaviour in done_experiments.keys():
        for generation_method in train_gen_methods:
            if behaviour in ["deception", "sandbagging"] and generation_method == "on_policy":
                continue
            
            ds = list(done_experiments[behaviour].keys())[1:]
            datasource_ID, datasource_OOD = (ds[0], ds[1])
            for datasource in [datasource_ID, datasource_OOD]:
                # Get experiments into format
                # [probe_type, behaviour, datasource, activations_model, generation_method, response_model, mode, cfg]
                off_policy_model = done_experiments[behaviour][datasource][activations_model]
                response_model = activations_model if generation_method != "off_policy" else off_policy_model
                train_setup = [[probe_type, behaviour, datasource, activations_model, generation_method, response_model, "train"]]
                
                test_setup = [[behaviour, datasource_ID, activations_model, test_gen_method, activations_model, "test"]]
                if done_experiments[behaviour]["test_both"]:
                    test_setup.append([behaviour, datasource_OOD, activations_model, test_gen_method, activations_model, "test"])
                do_single_probe_experiment(train_setup, test_setup)
            

def do_probe_experiment_combinations(
    new_probe_types, 
    new_behaviours, 
    new_activations_models,
    new_train_gen_methods, 
    new_test_gen_methods, 
    ):
    done_experiments = BEHAVIOUR_DATASOURCE_ACTMODEL_OFFPOLICYMODEL+BEHAVIOUR_DATASOURCE_ACTMODEL_OFFPOLICYMODEL_DECEPTION    
    # Iterate through all train and test pairs
    for probe_type in new_probe_types:
        for behaviour in new_behaviours:
            for activations_model in new_activations_models:
                for train_gen_method in new_train_gen_methods:
                    for test_gen_method in new_test_gen_methods:
                        if behaviour in ["deception", "sandbagging"] and (train_gen_method == "on_policy" or test_gen_method == "on_policy"):
                            continue
                        ds = list(done_experiments[behaviour].keys())[1:]
                        datasource_ID, datasource_OOD = (ds[0], ds[1])
                        for datasource in [datasource_ID, datasource_OOD]:
                            # Get experiments into format
                            # [probe_type, behaviour, datasource, activations_model, generation_method, response_model, mode, cfg]
                            off_policy_model = done_experiments[behaviour][datasource][activations_model]
                            response_model = activations_model if train_gen_method != "off_policy" else off_policy_model
                            train_setup = [[probe_type, behaviour, datasource, activations_model, train_gen_method, response_model, "train"]]
                            
                            test_setup = [[behaviour, datasource_ID, activations_model, test_gen_method, activations_model, "test"]]
                            if done_experiments[behaviour]["test_both"]:
                                test_setup.append([behaviour, datasource_OOD, activations_model, test_gen_method, activations_model, "test"])
                            do_single_probe_experiment(train_setup, test_setup)


if __name__ == "__main__":
    # MAKE SURE TO SET HF_TOKEN IN COMMAND LINE AND WANDB KEY AT TOP OF THIS FILE
    
    # Option 1: Set probe type and activation model and keep all other parameters same as in initial experiments
    for test_incentivised in [False, True]:
        do_probe_experiment_default(
            probe_type = ["mean", "attention_torch"][1],
            activations_model = ["llama_3b"][0], # currently the only option
            test_incentivised = test_incentivised, # False means we test against on policy not on policy incentivised data
        )
    
    # # Option 2: Set parameters based on all combinations
    # do_probe_experiment_combinations(
    #     new_probe_types = ["attention_torch", "mean"],
    #     new_behaviours = ["refusal", "lists", "metaphors", "science", "sycophancy", "authority", "deception", "sandbagging"],
    #     new_activations_models = ["llama_3b", "qwen_3b"],
    #     new_train_gen_methods = ["on_policy", "incentivised", "prompted", "off_policy"],
    #     new_test_gen_methods = ["on_policy", "incentivised", "prompted", "off_policy"],
    # )

    # # Option 3: Set experiments manually
    # # [probe_type, behaviour, datasource, activations_model, generation_method, response_model, mode, cfg]
    # train_setup = [
    #     ["mean", "refusal", "rlhf", "llama_3b", "on_policy", "llama_3b", "train"],
    #     ["mean", "refusal", "jailbreaks", "llama_3b", "on_policy", "llama_3b", "train"],
    # ]
    # test_setup = [
    #     ["refusal", "rlhf", "llama_3b", "on_policy", "llama_3b", "test"],
    #     ["refusal", "jailbreaks", "llama_3b", "on_policy", "llama_3b", "test"],
    # ]
    # for tr in train_setup:
    #     do_single_probe_experiment([tr], test_setup)