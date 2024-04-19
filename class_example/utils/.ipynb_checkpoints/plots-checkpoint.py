"""
Purpose: Plotting Tools
"""


import numpy as np
import matplotlib.pyplot as plt


plt.style.use("ggplot")


def plot_comparisons(all_results, figsize=(8, 4), fontsize=14):
    """
    Purpose:
    - Visualize comparison analytics
    
    Arguments:
    - all_results (np.ndarray[float]): comparisons between predictions and ground truths
    - figsize (tuple[int]): size of figure organized width by height
    - fontsize (int): size of font
    """

    fig, ax = plt.subplots(figsize=figsize)

    ax.boxplot(all_results)

    ax.set_xlabel("Features", fontsize=fontsize)
    ax.set_ylabel("Measure", fontsize=fontsize)
    ax.set_title("Prediction Analysis", fontsize=fontsize)
    ax.set_ylim([-0.05, 1.05])

    fig.tight_layout()


def plot_training(x_vals, y_vals, title, x_label, y_label, 
                  figsize=(8, 4), fontsize=14):
    """
    Purpose:
    - Visualize training analytics
    
    Arguments:
    - x_vals (list[any]): time measurement (i.e., steps, epochs)
    - y_vals (np.ndarray[float]): analytics to show (i.e., training error)
    - title (str): title of plot
    - x_label (str): label of x-axis
    - y_label (str): label of y-axis
    - figsize (tuple[int]): size of figure organized width by height
    - fontsize (int): size of font
    """

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x_vals, y_vals, linewidth=5)

    ax.set_title("%s" % title, fontsize=fontsize)
    ax.set_xlabel("%s" % x_label, fontsize=fontsize)
    ax.set_ylabel("%s" % y_label, fontsize=fontsize)

    fig.tight_layout()


def show_examples(dataset, task, figsize=(8, 4), fontsize=14):
    """
    Purpose:
    - Plot example from sequence dataset
    
    Arguments:
    - dataset (Dataset): machine learning dataset
    - tag (str): plot title
    - figsize (tuple[int]): size of figure organized width by height
    - fontsize (int): size of font
    """

    index = np.random.randint(len(dataset))

    fig, ax = plt.subplots(figsize=figsize)

    for i, seq_ele in enumerate(dataset.samples[index]):

        freq = dataset.labels[index][0]
        
        x_vals = np.linspace(0, 1 / freq, len(seq_ele))
        
        ax.plot(x_vals, seq_ele, label="Sequence %s" % (i + 1))

        ax.set_xlabel("Time", fontsize=fontsize)
        ax.set_ylabel("Measure", fontsize=fontsize)

        if task == 0:
            tag = "Many To One, Frequency = %s" % freq
        else:
            p_shift = dataset.labels[index][1]
            tag = "Many To Many, Frequency = %s, Phase = %s" % (freq, p_shift)

        ax.set_title(tag , fontsize=fontsize)
        
        ax.legend()
    
    fig.tight_layout()


def show_dataset(params, dataset):
    """
    Purpose:
    - Visualize dataset given experiment objective
    
    Arguments:
    - params (dict[str]): user defined parameters
    - dataset (Dataset): machine learning dataset
    
    Returns:
    - (Function): visualization function
    """
    
    task = params["experiment"]

    if task == 0 or task == 1:
        visualize = show_examples
    elif task == 2:
        visualize = show_forcasting
    else:
        raise NotImplementedError

    return visualize(dataset, task)