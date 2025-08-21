import probe_gen.probes as probes
from probe_gen.probes import save_probe_dict_results


def run_grid_experiment(dataset_names, layer, use_bias_list, normalize_list, C_list, activations_model):

    train_datasets = {}
    val_datasets = {}
    test_datasets = {}
    for dataset_name in dataset_names:
        activations_tensor, attention_mask, labels_tensor = probes.load_hf_activations_and_labels_at_layer(dataset_name, layer, verbose=True)
        activations_tensor = probes.MeanAggregation()(activations_tensor, attention_mask)
        train_dataset, val_dataset, test_dataset = probes.create_activation_datasets(activations_tensor, labels_tensor, val_size=0, test_size=0.2, balance=True, verbose=True)
        train_datasets[dataset_name] = train_dataset
        val_datasets[dataset_name] = val_dataset
        test_datasets[dataset_name] = test_dataset


    for train_index in len(dataset_names):
        train_dataset_name = dataset_names[train_index]
        # Initialise and fit a probe with the dataset
        probe = probes.SklearnLogisticProbe(use_bias=use_bias_list[train_index], normalize=normalize_list[train_index], C=C_list[train_index])
        probe.fit(train_datasets[train_dataset_name], val_datasets[train_dataset_name])

        for test_dataset_name in dataset_names:

            eval_dict, _, _ = probe.eval(test_datasets[test_dataset_name])
            save_probe_dict_results(eval_dict, "mean", use_bias_list[train_index], normalize_list[train_index], C_list[train_index], layer, train_dataset_name, test_dataset_name, activations_model)

    

