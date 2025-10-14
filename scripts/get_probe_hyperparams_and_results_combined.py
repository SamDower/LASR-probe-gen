# Standard library imports
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import torch
from huggingface_hub import login

import probe_gen.probes as probes
from probe_gen.config import (
    BEHAVIOUR_DATASOURCE_ACTMODEL_OFFPOLICYMODEL,
    BEHAVIOUR_DATASOURCE_ACTMODEL_OFFPOLICYMODEL_DECEPTION,
    ConfigDict,
)
from probe_gen.standard_experiments.hyperparameter_search import (
    get_best_hyperparams_for_train_setup,
    load_best_params_from_search,
    pick_popular_hyperparam,
    run_full_hyp_search_on_layers,
)

# If getting 'Could not find project LASR_probe_gen' get key from https://wandb.ai/authorize and paste below
os.environ["WANDB_SILENT"] = "true"
import wandb

wandb.login(key="")


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
    if tr[0][1] in ["deception", "sandbagging"]:
        generation_methods = ["incentivised", "prompted"]
    else:
        generation_methods = ["on_policy", "incentivised"]
    for generation_method in  generation_methods:
        if tr[0][0] == "mean":
            # Search all layers for mean probes
            layers_list = [6,9,12,15,18,21]
        elif tr[0][0] == "attention_torch":
            # Search only best mean probe layer for attention probes, or 12 if no mean probe layer found
            cfg = ConfigDict.from_json(tr[0][3], "mean", tr[0][1])
            if cfg is None or "layer" not in cfg:
                layers_list = [12]
            else:
                layers_list = [cfg.layer]
        else:
            raise ValueError(f"Probe type {tr[0][0]} not supported")
        
        run_full_hyp_search_on_layers(
            tr[0][0], tr[0][1], tr[0][2], tr[0][3], tr[0][3], generation_method, "train", layers_list
        )
        
        best_params_list.append(load_best_params_from_search(
            tr[0][0], tr[0][1], tr[0][2], generation_method, tr[0][3], tr[0][3], layers_list
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


def train_on_combined_distributions(
    probe_type, 
    behaviour, 
    datasources, 
    activations_model, 
    generation_methods, 
    test_setup,
    response_model=None,
    save_locally=True
):
    """
    Trains a single probe on combined data from multiple distributions.
    
    Args:
        probe_type: "mean" or "attention_torch"
        behaviour: e.g., "refusal", "deception"
        datasources: list like ["rlhf", "jailbreaks"] or ["writingprompts", "ultrachat"]
        activations_model: e.g., "llama_3b"
        generation_methods: list like ["on_policy", "incentivised"]
        test_setup: list of test configs [[behaviour, datasource, activations_model, generation_method, response_model, mode], ...]
        response_model: defaults to activations_model if None
        save_locally: if True, save results to JSON file as backup
    """
    if response_model is None:
        response_model = activations_model
    
    # Clean up any existing WandB runs before starting
    try:
        if wandb.run is not None:
            wandb.finish()
            time.sleep(1)
    except Exception:
        pass
    
    # Load hyperparams for each distribution and inspect
    print("=" * 50)
    print("Loading hyperparameters for each distribution...")
    print("=" * 50)
    configs = []
    for i, (datasource, gen_method) in enumerate(zip(datasources, generation_methods)):
        train_setup = [[probe_type, behaviour, datasource, activations_model, 
                        gen_method, response_model, "train"]]
        train_setup = get_best_hyperparams_for_train_setup(train_setup)
        cfg = train_setup[0][7]
        configs.append(cfg)
        print(f"\nDistribution {i+1}: {datasource} + {gen_method}")
        print(f"  Config: {cfg}")
    
    # Check if configs match
    print("\n" + "=" * 50)
    if all(str(c) == str(configs[0]) for c in configs):
        print("✓ All distributions have same hyperparameters!")
    else:
        print("⚠ WARNING: Distributions have different hyperparameters!")
        print("  Using hyperparameters from first distribution.")
    print("=" * 50 + "\n")
    
    # Use config from first distribution
    cfg = configs[0]
    
    # Load and combine all training data
    print("Loading and combining training data...")
    print("(Note: First-time download from HuggingFace can take 3-10 minutes)")
    all_activations = []
    all_labels = []
    
    for i, (datasource, gen_method) in enumerate(zip(datasources, generation_methods), 1):
        print(f"\n  [{i}/{len(datasources)}] Loading: {datasource} + {gen_method}")
        print("      Downloading/loading activations from HuggingFace...")
        
        start_time = time.time()
        acts, mask, labels = probes.load_hf_activations_at_layer(
            behaviour, datasource, activations_model, response_model, 
            gen_method, "train", cfg.layer, and_labels=True, verbose=True
        )
        elapsed = time.time() - start_time
        print(f"      ✓ Loaded in {elapsed:.1f}s")
        
        if "mean" in probe_type:
            print("      Applying mean aggregation...")
            acts = probes.MeanAggregation()(acts, mask)
            print("      ✓ Aggregation complete")
        
        all_activations.append(acts)
        all_labels.append(labels)
        print(f"      Shape: {acts.shape}, Labels: {labels.shape}")
        print(f"      Memory: {acts.element_size() * acts.nelement() / 1024**2:.1f} MB")
    
    # Concatenate
    print("\n" + "=" * 50)
    print("Concatenating all datasets...")
    print("=" * 50)
    combined_acts = torch.cat(all_activations, dim=0)
    combined_labels = torch.cat(all_labels, dim=0)
    total_samples = len(combined_acts)
    print(f"✓ Combined data shape: {combined_acts.shape}")
    print(f"✓ Combined labels shape: {combined_labels.shape}")
    print(f"✓ Total samples: {total_samples}")
    print(f"✓ Total memory: {combined_acts.element_size() * combined_acts.nelement() / 1024**2:.1f} MB")
    
    # Manual split with shuffling
    print("\n" + "=" * 50)
    print("Creating train/val split...")
    print("=" * 50)
    
    # Shuffle indices
    print("Shuffling indices...")
    indices = torch.randperm(total_samples)
    train_end = total_samples - 1000
    val_end = train_end + 500
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    print(f"✓ Shuffled {total_samples} samples")
    
    # Create dictionary-style datasets (compatible with sklearn probes)
    print("Creating train/val datasets...")
    train_dataset = {
        'X': combined_acts[train_indices],
        'y': combined_labels[train_indices]
    }
    val_dataset = {
        'X': combined_acts[val_indices],
        'y': combined_labels[val_indices]
    }
    
    train_size = len(train_dataset['X'])
    val_size = len(val_dataset['X'])
    
    print(f"✓ Dataset sizes - Train: {train_size}, Val: {val_size}")
    assert train_size >= 1000, f"Train dataset too small: {train_size} samples"
    
    # Check label distribution
    train_pos = (train_dataset['y'] == 1).sum().item()
    train_neg = (train_dataset['y'] == 0).sum().item()
    print(f"✓ Train label distribution - Positive: {train_pos}, Negative: {train_neg} (balance: {train_pos/train_size:.2%})")
    
    # Train probe
    print("\n" + "=" * 50)
    print(f"Training {probe_type} probe...")
    print("=" * 50)
    print(f"Hyperparameters: {dict(cfg)}")
    
    if probe_type == "attention_torch":
        probe = probes.TorchAttentionProbe(cfg)
    elif probe_type == "mean_torch":
        probe = probes.TorchLinearProbe(cfg)
    elif probe_type == "mean":
        probe = probes.SklearnLogisticProbe(cfg)
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
    
    print("Starting training...")
    train_start = time.time()
    probe.fit(train_dataset, val_dataset)
    train_elapsed = time.time() - train_start
    print(f"✓ Training complete in {train_elapsed:.1f}s!")
    
    # Evaluate on all test setups
    print("\n" + "=" * 50)
    print("Evaluating on test sets...")
    print("=" * 50)
    
    all_results = []
    
    for j, ts in enumerate(test_setup):
        print(f"\n[Test {j+1}/{len(test_setup)}] {ts[1]} + {ts[3]}")
        
        # Load test data
        print("  Loading test activations...")
        test_load_start = time.time()
        activations_tensor, attention_mask, labels_tensor = probes.load_hf_activations_at_layer(
            ts[0], ts[1], ts[2], ts[4], ts[3], ts[5], cfg.layer, and_labels=True, verbose=True
        )
        test_load_elapsed = time.time() - test_load_start
        print(f"  ✓ Loaded in {test_load_elapsed:.1f}s")
        
        if "mean" in probe_type:
            print("  Applying mean aggregation...")
            activations_tensor = probes.MeanAggregation()(activations_tensor, attention_mask)
        
        # Create test dataset
        if ts[1] == "jailbreaks":
            _, _, test_dataset = probes.create_activation_datasets(
                activations_tensor, labels_tensor, splits=[3500, 500, 1000]
            )
        elif ts[1] == "trading":
            _, _, test_dataset = probes.create_activation_datasets(
                activations_tensor, labels_tensor, splits=[2500, 500, 500]
            )
        elif ts[0] in ["deception", "sandbagging"]:
            _, _, test_dataset = probes.create_activation_datasets(
                activations_tensor, labels_tensor, splits=[0, 0, 500]
            )
        else:
            _, _, test_dataset = probes.create_activation_datasets(
                activations_tensor, labels_tensor, splits=[0, 0, 1000]
            )
        
        # Evaluate
        print("  Running evaluation...")
        eval_start = time.time()
        eval_dict, _, _ = probe.eval(test_dataset)
        eval_elapsed = time.time() - eval_start
        
        # Convert numpy types to native Python types for JSON serialization
        eval_dict_serializable = {}
        for k, v in eval_dict.items():
            if hasattr(v, 'item'):  # numpy scalar
                eval_dict_serializable[k] = v.item()
            else:
                eval_dict_serializable[k] = v
        
        print(f"  ✓ Evaluation complete in {eval_elapsed:.1f}s")
        print(f"  Results: {eval_dict_serializable}")
        
        # Store results
        all_results.append((ts, eval_dict, eval_dict_serializable))
    
    # Create descriptive train_set name
    combined_train_desc = "+".join([f"{ds}_{gm}" for ds, gm in zip(datasources, generation_methods)])
    
    # Save results locally if requested
    if save_locally:
        print("\n" + "=" * 50)
        print("Saving results locally...")
        print("=" * 50)
        
        results_dir = Path("combined_probe_results")
        results_dir.mkdir(exist_ok=True)
        
        for j, (ts, eval_dict, eval_dict_serializable) in enumerate(all_results):
            result_data = {
                'probe_type': probe_type,
                'behaviour': behaviour,
                'train_set': {
                    'datasources': datasources,
                    'generation_methods': generation_methods,
                    'combined_name': combined_train_desc,
                    'activations_model': activations_model,
                    'response_model': response_model
                },
                'test_set': {
                    'behaviour': ts[0],
                    'datasource': ts[1],
                    'activations_model': ts[2],
                    'generation_method': ts[3],
                    'response_model': ts[4],
                    'split': ts[5]
                },
                'hyperparams': dict(cfg),
                'results': eval_dict_serializable,
                'train_size': train_size,
                'val_size': val_size
            }
            
            filename = f"{combined_train_desc}_test-{ts[1]}_{ts[3]}.json"
            filepath = results_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            print(f"  ✓ Saved to {filepath}")
    
    # Try to save to WandB (but don't worry if it fails)
    print("\n" + "=" * 50)
    print("Attempting to save results to WandB...")
    print("=" * 50)
    
    successful_saves = 0
    
    for j, (ts, eval_dict, _) in enumerate(all_results):
        if "torch" in probe_type:
            hyperparams = [cfg.layer, cfg.use_bias, cfg.normalize, cfg.lr, cfg.weight_decay]
        elif probe_type == "mean":
            hyperparams = [cfg.layer, cfg.use_bias, cfg.normalize, cfg.C]
        
        # Ensure clean WandB state
        try:
            if wandb.run is not None:
                wandb.finish()
                time.sleep(0.5)
        except Exception:
            pass
        
        try:
            probes.wandb_interface.save_probe_dict_results(
                eval_dict=eval_dict,
                probe_type=probe_type,
                behaviour=behaviour,
                train_set=[combined_train_desc, "combined", response_model],
                test_set=[ts[1], ts[3], ts[4]],
                activations_model=activations_model,
                hyperparams=hyperparams,
            )
            print(f"  ✓ Saved to WandB for test {j+1}: {ts[1]} + {ts[3]}")
            successful_saves += 1
            
            # Clean up after successful save
            try:
                if wandb.run is not None:
                    wandb.finish()
                    time.sleep(0.5)
            except Exception:
                pass
                
        except Exception as e:
            print(f"  ⚠ WandB logging failed for test {j+1} (results saved locally): {str(e)[:100]}")
            print(f"  Full error: {str(e)}")  # Remove [:100] to see full error
            import traceback
            traceback.print_exc()  # Print full stack trace
            try:
                if wandb.run is not None:
                    wandb.finish()
            except Exception:
                pass
    
    if successful_saves > 0:
        print(f"\n✓ Successfully saved {successful_saves}/{len(all_results)} results to WandB")
    else:
        print("\n⚠ WandB logging unavailable - all results saved locally in 'combined_probe_results/' directory")
    
    # Final cleanup
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass
    
    print("\n✓ All evaluations complete!")
    return probe, all_results


def do_combined_probe_experiment(
    probe_type,
    behaviour,
    datasources,
    activations_model,
    generation_methods,
    test_setup,
    response_model=None
):
    """
    Runs a combined probe experiment on multiple datasets.
    
    Args:
        probe_type: "mean" or "attention_torch"
        behaviour: e.g., "refusal", "lists", etc.
        datasources: list of datasources to combine, e.g., ["writingprompts", "ultrachat"]
        activations_model: e.g., "llama_3b"
        generation_methods: list of generation methods corresponding to datasources
        test_setup: list of test configs
        response_model: defaults to activations_model if None
    """
    print(f"Processing {probe_type} for {behaviour} combining {datasources} with {generation_methods}")
    
    # Train on combined datasets and evaluate
    probe, results = train_on_combined_distributions(
        probe_type=probe_type,
        behaviour=behaviour,
        datasources=datasources,
        activations_model=activations_model,
        generation_methods=generation_methods,
        test_setup=test_setup,
        response_model=response_model,
        save_locally=False
    )
    
    # Delete activation files to save space
    hf_home = os.path.expanduser("~/.hf_home")
    if os.path.exists(hf_home):
        print("Deleting activation files")
        shutil.rmtree(hf_home)
    
    return probe, results


def do_combined_probe_experiment_default(probe_type, activations_model, behaviour, datasource_pairs, test_datasources):
    """
    Run combined probe experiments using default settings.
    
    Args:
        probe_type: "mean" or "attention_torch"
        activations_model: e.g., "llama_3b"
        behaviour: e.g., "lists", "refusal"
        datasource_pairs: list of tuples, e.g., [("writingprompts", "ultrachat")]
        test_datasources: list of datasources to test on, e.g., ["shakespeare"]
    """
    done_experiments = BEHAVIOUR_DATASOURCE_ACTMODEL_OFFPOLICYMODEL
    done_experiments.update(BEHAVIOUR_DATASOURCE_ACTMODEL_OFFPOLICYMODEL_DECEPTION)
    
    # Determine generation methods based on behaviour
    if behaviour in ["deception", "sandbagging"]:
        train_gen_methods = ['incentivised', 'prompted', 'off_policy']
    else:
        train_gen_methods = ['on_policy', 'incentivised', 'prompted', 'off_policy']
    
    # Iterate through all generation methods
    for generation_method in train_gen_methods:
        for datasource_pair in datasource_pairs:
            # Both datasets use the same generation method
            generation_methods = [generation_method] * len(datasource_pair)
            
            # Build test setup
            test_setup = []
            for test_datasource in test_datasources:
                test_gen_method = 'on_policy' if behaviour not in ["deception", "sandbagging"] else 'incentivised'
                test_setup.append([behaviour, test_datasource, activations_model, test_gen_method, activations_model, "test"])
            
            do_combined_probe_experiment(
                probe_type=probe_type,
                behaviour=behaviour,
                datasources=list(datasource_pair),
                activations_model=activations_model,
                generation_methods=generation_methods,
                test_setup=test_setup,
                response_model=activations_model
            )


if __name__ == "__main__":
    # MAKE SURE TO SET HF_TOKEN IN COMMAND LINE AND WANDB KEY AT TOP OF THIS FILE
    
    # =============================================================================
    # OPTION 1: Run all generation methods for a behaviour (recommended)
    # =============================================================================
    # This will run on_policy, incentivised, prompted, and off_policy automatically
    # do_combined_probe_experiment_default(
    #     probe_type="mean",
    #     activations_model="llama_3b",
    #     behaviour="lists",
    #     datasource_pairs=[("writingprompts", "ultrachat")],
    #     test_datasources=["shakespeare"]
    # )

    # Experiment 1: Train on writingprompts + ultrachat, test on ultrachat (ID)
    do_combined_probe_experiment_default(
        probe_type="mean",
        activations_model="llama_3b",
        behaviour="lists",
        datasource_pairs=[("writingprompts", "ultrachat")],
        test_datasources=["ultrachat"]
    )
    
    # Experiment 2: Train on writingprompts + shakespeare, test on ultrachat (OOD)
    do_combined_probe_experiment_default(
        probe_type="mean",
        activations_model="llama_3b",
        behaviour="lists",
        datasource_pairs=[("writingprompts", "shakespeare")],
        test_datasources=["ultrachat"]
    )
    
    # =============================================================================
    # OPTION 2: Run a single generation method only
    # =============================================================================
    # Uncomment this to run just one specific generation method
    # probe, results = do_combined_probe_experiment(
    #     probe_type="mean",
    #     behaviour="lists",
    #     datasources=["writingprompts", "ultrachat"],
    #     activations_model="llama_3b",
    #     generation_methods=["on_policy", "on_policy"],  # Both datasets use same method
    #     test_setup=[
    #         ["lists", "shakespeare", "llama_3b", "on_policy", "llama_3b", "test"],
    #     ],
    #     response_model="llama_3b"
    # )
    
    # =============================================================================
    # OPTION 3: Different behaviours (examples for when you have the datasets)
    # =============================================================================
    
    # Example 1: Refusal behaviour
    # do_combined_probe_experiment_default(
    #     probe_type="mean",
    #     activations_model="llama_3b",
    #     behaviour="refusal",
    #     datasource_pairs=[("rlhf", "jailbreaks")],
    #     test_datasources=["rlhf", "jailbreaks"]  # Can test on both ID and OOD
    # )
    
    # Example 2: Sycophancy behaviour
    # do_combined_probe_experiment_default(
    #     probe_type="mean",
    #     activations_model="llama_3b",
    #     behaviour="sycophancy",
    #     datasource_pairs=[("dataset1", "dataset2")],  # Replace with your datasets
    #     test_datasources=["ood_dataset"]
    # )
    
    # Example 3: Science behaviour
    # do_combined_probe_experiment_default(
    #     probe_type="mean",
    #     activations_model="llama_3b",
    #     behaviour="science",
    #     datasource_pairs=[("dataset1", "dataset2")],
    #     test_datasources=["ood_dataset"]
    # )
    
    # Example 4: Deception behaviour (note: uses different generation methods automatically)
    # do_combined_probe_experiment_default(
    #     probe_type="mean",
    #     activations_model="llama_3b",
    #     behaviour="deception",
    #     datasource_pairs=[("dataset1", "dataset2")],
    #     test_datasources=["ood_dataset"]
    # )
    
    # =============================================================================
    # OPTION 4: Multiple test datasets
    # =============================================================================
    # probe, results = do_combined_probe_experiment(
    #     probe_type="mean",
    #     behaviour="lists",
    #     datasources=["writingprompts", "ultrachat"],
    #     activations_model="llama_3b",
    #     generation_methods=["on_policy", "on_policy"],
    #     test_setup=[
    #         ["lists", "shakespeare", "llama_3b", "on_policy", "llama_3b", "test"],
    #         ["lists", "writingprompts", "llama_3b", "on_policy", "llama_3b", "test"],
    #         ["lists", "ultrachat", "llama_3b", "on_policy", "llama_3b", "test"],
    #     ],
    #     response_model="llama_3b"
    # )

