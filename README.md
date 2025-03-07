# Optimizing Piecewise-Constant Objectives
Public code to accompany Section 6 of my thesis. This repo contains:
- Code for Girard Accuracy and Gaussian Mixture Half-space Pruning (`piecewise_constant_objectives/algorithms.py`)
- Code to load our cross-entropy-trained RNNs of various sizes (`piecewise_constant_objectives/model.py`)
- Code for training a RNN against GMHP, SS, CE, or Hook loss (`piecewise_constant_objectives/optimization.py`)
- Code for generating many of loss curves in the thesis (`plots.ipynb`). Also available as a Colab notebook here: https://colab.research.google.com/github/GabrielDWu/piecewise-constant-objectives/blob/main/plots.ipynb

# Installation instructions
Clone this repo and install with `pip install -e PATH/TO/THIS/REPO`

# Public datafiles
All loss curves are available at `gs://arc-ml-public/gabe-thesis/all_losses_and_accs.json`