import probe_gen.probes as probes
from probe_gen.probes import save_probe_dict_results


def run_grid_experiment(dataset_names, layer, use_bias, normalize):

    train_datasets = {}
    val_datasets = {}
    test_datasets = {}
    for dataset_name in dataset_names:
        activations_tensor, attention_mask, labels_tensor = probes.load_hf_activations_and_labels_at_layer(dataset_name, layer, verbose=True)
        train_dataset, val_dataset, test_dataset = probes.create_activation_datasets(activations_tensor, attention_mask, labels_tensor, val_size=0, test_size=0.2, verbose=True)
        train_datasets[dataset_name] = train_dataset
        val_datasets[dataset_name] = val_dataset
        test_datasets[dataset_name] = test_dataset


    for train_dataset_name in dataset_names:
        # Initialise and fit a probe with the dataset
        probe = probes.SklearnMeanLogisticProbe(use_bias=use_bias)
        probe.fit(train_datasets[train_dataset_name], val_datasets[train_dataset_name], normalize=normalize)

        for test_dataset_name in dataset_names:

            eval_dict, _, _ = probe.eval(test_datasets[test_dataset_name])
            save_probe_dict_results(eval_dict, "mean", use_bias, normalize, layer, train_dataset_name, test_dataset_name)

    

