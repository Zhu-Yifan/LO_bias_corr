# LO_bias_corr

This repository contains the code used in the manuscript entitled "A Neural Network Approach for Bias Correction of High-Resolution Ocean Biogeochemical Hindcast Fields", which develops and applies a neural network framework to correct dissolved oxygen (DO) biases in a three-decade (1993–2022), high-resolution regional ocean biogeochemical hindcast generated using the LiveOcean modeling system. 

## Workflow

### (1) Create a python environment

Assuming anaconda or miniconda is already installed, create a python environment by executing:

```bash
conda env create -f bias.yml
conda activate bias
```
The `bias.yml` file specifies the Python packages and dependencies required to reproduce this work, including TensorFlow, scikit-learn, xarray, and NumPy, etc.

### (2) Data preparation
Relevant notebooks are provided in the [`data`](./data) folder.

A multidecadal collection of dissolved oxygen (DO) observations along the U.S. Pacific Northwest coast, spanning 1993–2022, was compiled and used to quantify and characterize DO biases in the LiveOcean hindcast.


### (3) Training, hyperparameter tuning, evaluation

See the [`training_and_tuning`](./training_and_tuning) folder for code related to data splitting, neural network construction, hyperparameter tuning, and model evaluation.

### (4) Application 

Apply the trained neural network to the full model fields, individual model layers, or mooring locations, and calculate and visualize hypoxia metrics.

Relevant notebooks are provided in the [`application`](./application) folder 



