def get_params(method="default"):

    params = {}
    
    if method == "grid_search":
        # Grid Search Configuration
        params["config"] = {"hidden_size": [64, 128, 1024],
                            "learning_rate": [1e-5],
                            "batch_size": [32],
                            "num_layers": [4],
                            "max_epochs": [100] }
    elif method == "default":
        # Default Configuration
        params['config'] = {"hidden_size": 64,
                            "learning_rate": 1e-2,
                            "batch_size": 100,
                            "num_layers": 4,
                            "max_epochs": 250 }
    else:
        raise ValueError("Invalid method. Must be either 'grid_search' or 'default'.")

    # Experiment Objective
    # - experiment: (0) Many to 1, (1) many to many

    params["experiment"] = 0

    # System Experiemnts

    params["system"] = {"accelerator": "gpu", 
                        "strategy": "auto", 
                        "num_devices": 1,
                        "num_workers": 0}

    # Path Parameters
    # - results: path to store performance analytics
    # - version: version number of experiment to investigate for results

    params["paths"] = {"results": "results",
                    "version": 0}
    
    # Evaluation Parameters
    # - tags: column names from saved algorithm training analytics

    params["evaluation"] = {"tags": ["train_error_epoch", 
                                    "valid_error_epoch", 
                                    "lr-Adam"]}

    # Data Parameters

    params["data"] = {"num_sequence": 25,
                    "num_samples": 10000,
                    "num_features": 18 }
    
    return params