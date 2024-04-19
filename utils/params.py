def get_params():

    params = {}

    # Create: Experiment Objective
    # - experiment: (0) Many to 1, (1) many to many

    params["experiment"] = 0

    # Create: System Experiemnts
    # - accelerator: gpu type, (MacOS-M, NVIDIA) gpu, (N/A) cpu
    # - strategy: type of distributed computation environment
    # - num_devices: number of gpus available
    # - num_workers: number of cpus for dataloader

    params["system"] = {"accelerator": "gpu", 
                        "strategy": "auto", 
                        "num_devices": 1,
                        "num_workers": 8}

    # Create: Path Parameters
    # - results: path to store performance analytics
    # - version: version number of experiment to investigate for results

    params["paths"] = {"results": "results",
                    "version": 0}

    # Create: Data Parameters
    # - num_sequence: sequence size for network observations
    # - num_features: number of dataset features 
    # - num_samples: number of dataset observations
    # - amplitude: min and max range for signal height
    # - frequency: min and max range for signal repitition

    params["data"] = {"num_sequence": 3,
                    "num_samples": 10000,
                    "num_features": 18,
                    "amplitude": {"min": 1, "max": 10},
                    "frequency": {"min": 1, "max": 10}}

    # Create: Model Parameters
    # - batch_size: number of observations per sample to network
    # - num_layers: number of stacked lstm cells
    # - hidden_layers: number of lstm features
    # - learning_rate: how much to listen to gradient
    # - num_epochs: number of times model observes full dataset 

    params["model"] = {"batch_size": 64,
                    "num_layers": 3,
                    "hidden_size": 512,
                    "learning_rate": 3e-3,
                    "num_epochs": 100,}

    # Create: Evaluation Parameters
    # - tags: column names from saved algorithm training analytics

    params["evaluation"] = {"tags": ["train_error_epoch", 
                                    "valid_error_epoch", 
                                    "lr-Adam"]}
    
    return params