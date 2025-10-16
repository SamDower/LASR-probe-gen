import sys
from pathlib import Path

from collections import Counter
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
import probe_gen.probes as probes
from probe_gen.config import ConfigDict
from probe_gen.probes.wandb_interface import load_probe_eval_dicts_as_df

LAYERS_LIST = [6,9,12,15,18,21]
USE_BIAS_RANGE = [True]
NORMALIZE_RANGE = [True]
C_RANGE = [0.001, 0.01, 0.1, 1, 10]
LR_RANGE = [1e-4, 1e-3, 1e-2]
WEIGHT_DECAY_RANGE = [0, 1e-5, 1e-4]


def run_full_hyp_search_on_layers(
    probe_type: str, 
    behaviour: str, 
    datasource: str, 
    activations_model: str="llama_3b", 
    response_model: str="llama_3b", 
    generation_method: str="on_policy", 
    mode: str="train", 
    layers_list: list=LAYERS_LIST):
    
    best_auroc = 0
    best_params = []
    # Do initial search on everything except normalize and use bias
    norm_bias_params = [True, True]
    normalize = norm_bias_params[0]
    use_bias = norm_bias_params[1]
    for layer in layers_list:
        print(f"#### #### #### Evaluating layer {layer}")
        activations_tensor, attention_mask, labels_tensor = probes.load_hf_activations_at_layer(
            behaviour, datasource, activations_model, response_model, generation_method, mode, layer, and_labels=True, verbose=False)
        if "mean" in probe_type:
            activations_tensor = probes.MeanAggregation()(activations_tensor, attention_mask)
        split_val = 2500 if behaviour in ["deception", "sandbagging"] else 3500
        split_val = 3000 if datasource == "shakespeare" else split_val
        train_dataset, val_dataset, _ = probes.create_activation_datasets(
            activations_tensor, labels_tensor, splits=[split_val, 500, 0], verbose=False)

        if 'torch' in probe_type:
            for lr in LR_RANGE:
                for weight_decay in tqdm(WEIGHT_DECAY_RANGE):
                    if probe_type == "mean_torch":
                        probe = probes.TorchLinearProbe(ConfigDict(use_bias=use_bias, normalize=normalize, lr=lr, weight_decay=weight_decay))
                    elif probe_type == "attention_torch":
                        probe = probes.TorchAttentionProbe(ConfigDict(use_bias=use_bias, normalize=normalize, lr=lr, weight_decay=weight_decay))
                    probe.fit(train_dataset, val_dataset, verbose=False)
                    eval_dict, _, _ = probe.eval(val_dataset)
                    if eval_dict['roc_auc'] > best_auroc:
                        best_auroc = eval_dict['roc_auc']
                        best_params = [layer, lr, weight_decay]
                    probes.wandb_interface.save_probe_dict_results(
                        eval_dict=eval_dict,
                        probe_type = probe_type,
                        behaviour = behaviour,
                        train_set=[datasource, generation_method, response_model],
                        test_set=[datasource, generation_method, response_model],
                        activations_model=activations_model,
                        hyperparams=[layer, use_bias, normalize, lr, weight_decay],
                    )
        
        elif probe_type == 'mean':
            for C in tqdm(C_RANGE):
                probe = probes.SklearnLogisticProbe(ConfigDict(use_bias=use_bias, C=C, normalize=normalize))
                probe.fit(train_dataset, val_dataset, verbose=False)
                eval_dict, _, _ = probe.eval(val_dataset)
                if eval_dict['roc_auc'] > best_auroc:
                    best_auroc = eval_dict['roc_auc']
                    best_params = [layer, C]
                probes.wandb_interface.save_probe_dict_results(
                    eval_dict=eval_dict,
                    probe_type = probe_type,
                    behaviour = behaviour,
                    train_set=[datasource, generation_method, response_model],
                    test_set=[datasource, generation_method, response_model],
                    activations_model=activations_model,
                    hyperparams=[layer, use_bias, normalize, C],
                )
        
        else:
            print("Probe type not valid.")
            return

    # Do followup search on just whether to normalise and use bias
    if 'torch' in probe_type:
        layer, lr, weight_decay = best_params[0], best_params[1], best_params[2]
    elif probe_type == 'mean':
        layer, C = best_params[0], best_params[1]
    for use_bias in USE_BIAS_RANGE:
        for normalize in NORMALIZE_RANGE:
            if normalize == norm_bias_params[0] and use_bias == norm_bias_params[1]:
                continue
                
            if 'torch' in probe_type:
                if probe_type == "mean_torch":
                    probe = probes.TorchLinearProbe(ConfigDict(use_bias=use_bias, normalize=normalize, lr=lr, weight_decay=weight_decay))
                elif probe_type == "attention_torch":
                    probe = probes.TorchAttentionProbe(ConfigDict(use_bias=use_bias, normalize=normalize, lr=lr, weight_decay=weight_decay))
                hyperparams = [layer, use_bias, normalize, lr, weight_decay]
            elif probe_type == 'mean':
                probe = probes.SklearnLogisticProbe(ConfigDict(use_bias=use_bias, C=C, normalize=normalize))
                hyperparams = [layer, use_bias, normalize, C]
            
            probe.fit(train_dataset, None, verbose=False)
            eval_dict, _, _ = probe.eval(val_dataset)
            if eval_dict['roc_auc'] > best_auroc:
                best_auroc = eval_dict['roc_auc']
                norm_bias_params = [normalize, use_bias]
            
            probes.wandb_interface.save_probe_dict_results(
                eval_dict=eval_dict,
                probe_type = probe_type,
                behaviour = behaviour,
                train_set=[datasource, generation_method, response_model],
                test_set=[datasource, generation_method, response_model],
                activations_model=activations_model,
                hyperparams=hyperparams,
            )

    # Do followup search on just whether to normalise and use bias
    if 'torch' in probe_type:
        print(f"Best Params, Layer: {layer}, LR: {lr}, Weight Decay: {weight_decay}", end="")
    elif probe_type == 'mean':
        print(f"Best Params, Layer: {layer}, C: {C}", end="")
    print(f", Normalize: {norm_bias_params[0]}, Use Bias: {norm_bias_params[1]}")
    print(f"Best roc_auc: {best_auroc}")


def load_best_params_from_search(probe_type, behaviour, datasource, generation_method, response_model, activations_model, layers_list=LAYERS_LIST):
    df = load_probe_eval_dicts_as_df({
        "config.probe/type": probe_type,
        "config.behaviour": behaviour,
        "config.train/datasource": datasource,
        "config.train/generation_method": generation_method,
        "config.train/response_model": response_model,
        "config.test/datasource": datasource,
        "config.test/generation_method": generation_method,
        "config.test/response_model": response_model,
        "config.activations_model": activations_model,
        "state": "finished"  # Only completed runs
    })
    if df.shape[0] == 0:
        raise ValueError(f"No runs found for {probe_type}, {behaviour}, {datasource}, {generation_method}, {response_model}, {activations_model}")
    
    best_auroc = 0
    best_params = {}
    for layer in layers_list:
        for use_bias in USE_BIAS_RANGE:
            for normalize in NORMALIZE_RANGE:
                if 'torch' in probe_type:
                    for lr in LR_RANGE:
                        for weight_decay in WEIGHT_DECAY_RANGE:
                            filtered_df = df[
                                (df['config_layer'] == layer) & 
                                (df['config_probe_normalize'] == normalize) & 
                                (df['config_probe_use_bias'] == use_bias) & 
                                (df['config_probe_lr'] == lr) & 
                                (df['config_probe_weight_decay'] == weight_decay)
                            ]
                            
                            if filtered_df.shape[0] >= 1:
                                roc_auc = filtered_df['metric_roc_auc'].iloc[-1]
                                if roc_auc > best_auroc:
                                    best_auroc = roc_auc
                                    best_params = filtered_df.iloc[-1].to_dict()
                    
                elif probe_type == 'mean':
                    for C in C_RANGE:
                        filtered_df = df[
                            (df['config_layer'] == layer) & 
                            (df['config_probe_normalize'] == normalize) & 
                            (df['config_probe_use_bias'] == use_bias) & 
                            (df['config_probe_C'] == C)
                        ]
                        
                        if filtered_df.shape[0] >= 1:
                            roc_auc = filtered_df['metric_roc_auc'].iloc[-1]
                            if roc_auc > best_auroc:
                                best_auroc = roc_auc
                                best_params = filtered_df.iloc[-1].to_dict()
                    
                else:
                    print("Probe type not valid.")
                    return

    best_params_format = {}
    for key in list(best_params.keys()):
        if key.startswith('config_probe_') and key != 'config_probe_type':
            best_params_format[key[len('config_probe_'):]] = best_params[key]
        elif key == 'config_layer':
            best_params_format[key[len('config_'):]] = best_params[key]
    print(f"Loaded Params: {best_params_format}")
    print(f"Loaded roc_auc: {best_auroc}")
    return best_params_format


def get_best_hyperparams_for_train_setup(train_setup):
    """
    Gets the best hyperparameters for the probes specified in the train_setup list.
    Args:
        train_setup (list): 
            [probe_type, behaviour, datasource, activations_model, generation_method, response_model, mode, cfg]
    """
    tr = train_setup
    
    # Get the best hyperparameters for each probe if not provided
    for i in range(len(tr)):
        if not isinstance(tr[i][-1], ConfigDict):
            best_cfg = None
            try:
                best_cfg = ConfigDict.from_json(tr[i][3], tr[i][0], tr[i][1])
            except KeyError:
                print(f"No best hyperparameters found for {tr[i][3]}, {tr[i][0]}, {tr[i][1]} locally, pulling from wandb...")
                best_cfg = load_best_params_from_search(tr[i][0], tr[i][1], tr[i][2], tr[i][4], tr[i][5], tr[i][3])
            if best_cfg is None or len(list(best_cfg.keys())) == 0:
                raise ValueError(f"No best hyperparameters found for {tr[i][3]}, {tr[i][0]}, {tr[i][1]}")
            
            if tr[i][0] == 'mean':
                tr[i].append(ConfigDict(layer=best_cfg.layer, use_bias=True, normalize=True, C=best_cfg.C))
            elif "torch" in tr[i][0]:
                tr[i].append(ConfigDict(layer=best_cfg.layer, use_bias=True, normalize=True, lr=best_cfg.lr, weight_decay=best_cfg.weight_decay))
            else:
                raise Exception(f"Wrong probe type: {tr[i][0]}")
    return tr


def pick_popular_hyperparam(best_params_list, param_name, params_range=None):
    """
    best_params_list: list of best hyperparameters for each policy
    param_name: name of the hyperparameter to pick (e.g. "layer", "c", "lr", "weight_decay")
    params_range: range of the hyperparameter to pick from (e.g. LAYERS_LIST, C_RANGE, LR_RANGE, WEIGHT_DECAY_RANGE)
    """
    if param_name == "layer" and params_range is None:
        params_range = LAYERS_LIST
    elif param_name == "c" and params_range is None:
        params_range = C_RANGE
    elif param_name == "lr" and params_range is None:
        params_range = LR_RANGE
    elif param_name == "weight_decay" and params_range is None:
        params_range = WEIGHT_DECAY_RANGE
    elif params_range is None:
        raise ValueError(f"Parameter name {param_name} not valid or params_range not provided.")
        
    n = len(best_params_list)
    counts = Counter(best_params_list)
    
    # Step 1: Check for majority element
    for num, count in counts.items():
        if count > n // 2:  # strictly more than half
            return num
    
    # Step 2: If no majority, take mean and find closest in params_range
    mean_val = sum(best_params_list) / n
    closest = min(params_range, key=lambda x: abs(x - mean_val))
    return closest
