def get_params(method="default"):
    """_summary_ This function creates a comprehensive set of hyperparameters for the LSTM or RNN model
    for the Room Occupancy dataset. The hyperparameters are used to train the model and evaluate its performance
    using PyTorch Lightning. The function can be used to either set the hyperparameters to default values or
    perform a grid search to find the optimal hyperparameters. The paper describes hyperparameters in detail
    and this function highlights some of the core aspects of the many tests conducted in this project to extend
    the original paper's focus on the more fundamental models - SVM, LDA.

    Args:
        method (str, optional): _description_. Defaults to "default". Optionally set to 'grid_search' to 
        seamlessly perform grid search per a set of specifications such as those displayed below.

    Raises:
        ValueError: _description_ Must be either 'grid_search' or 'default'.

    Returns:
        _type_: _description_ The hyperparameters for the LSTM or RNN model
    """
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
                            "max_epochs": 250 }
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
    
    # patience
    
    params["patience"] = 10
    
    # svm parameters
    
    params["svm"] = {"C": 100.0,
                    "kernel": "rbf" }
    
    return params