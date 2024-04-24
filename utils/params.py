def get_params(method="default"):

    params = {}
    
    if method == "grid_search":
        # Grid Search Configuration
        params["config"] = {"hidden_size": [128, 256],
                            "learning_rate": [0.001, 0.0001],
                            "batch_size": [64, 128],
                            "num_layers": [1, 3],
                            "max_epochs": [100] }
    elif method == "default":
        # Default Configuration
        params['config'] = {"hidden_size": 128,
                            "learning_rate": 0.0001,
                            "batch_size": 64,
                            "num_layers": 1,
                            "max_epochs": 100 }
    else:
        raise ValueError("Invalid method. Must be either 'grid_search' or 'default'.")

    # Experiment Objective
    # - experiment: (0) Many to 1, (1) many to many. (2) 1 to 1

    params["experiment"] = 0

    # System Experiemnts

    params["system"] = {"accelerator": "gpu", 
                        "strategy": "auto", 
                        "num_devices": 1,
                        "num_workers": 0}

    # Data Parameters

    params["data"] = {"num_sequence": 25,
                    "num_samples": 10000,
                    "num_features": 18 }
    
    return params