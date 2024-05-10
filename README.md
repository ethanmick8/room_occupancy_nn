# Room Occupancy Testing Codebase - Ethan Mick 

## Introduction  
This is the codebase for the final paper in Machine Learning and Pattern Recognition

## Requirements
Ensure you have Conda installed on your machine. You can download it from [Anaconda](https://www.anaconda.com/products/individual) or Miniconda from [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Miniconda is preferred and it is what was used here

## Installation

To set up your environment to run this library, follow these steps:

1. **Clone the Repository**
git clone https://github.com/ethanmick8/room_occupancy_nn.git  
cd room_occupancy_nn  
* OR, if you're viewing this directly from my submission, don't bother and just move to 2.  


2. **Create Conda Environment**
Create a Conda environment using the `environment.yml` file provided in the repository:  
```conda env create -f environment.yml```  
`requirements.txt` includes many of the core libraries in its list but it is not  
complete; this library is not designed to support packaging that would normally  
be expected when a file such as this is included because it's easier to just indicate  
the complete environment used as such.

3. **Activate the Environment**

## Getting Started

Here is a simple example of how to replicate experiments done with my codebase:
```python -m training.train --model_type svm --fold_index 10```  
The above runs training for an svm model utilizing 10-fold cross-validation  
Model specifics are always included in utils/params.py, i.e. for svm different  
tweaks such as the kernel used or the C value.
```python -m evaluation.evaluate --model_type svm --checkpoint_path ml_logs/models/svm/(experiment name) --cross_val 10```  
The above is the other runable command present in my library and it handles a lot.  
Obviously it performs evaluation, measuring model performance on classification,  
but it utilizes various plotting and calculation functions from utils/metrics.py to  
display lots of information on a model. It is dynamic in its ability to process cross  
validation training approaches as well as, exclusively for LSTM and SVM, traditional  
full approaches. View train.py and evaluate.py for further details on arguments and  
specific functionality.  
