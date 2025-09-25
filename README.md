# Multiscale Real-Time Inference of Nonlinear Embedding (MRINE)

MRINE is a multiscale dynamical model of multimodal time-series data that can have timescale differences and missing samples. MRINE accounts for these differences with its 
novel encoder design that consists of modality-specific encoder networks (MLP) and LDMs that learn within-modality dynamics. Then, a fusion network integrates information
across learned representations of each modality. Followed by a multiscale LDM, MRINE can perform both real-time and noncausal inference of multiscale latent factors in addition to
k-step-ahead prediction. After latent factors are extracted for the whole time horizon, they can be used for downstream prediction tasks such as behavior decoding for neuroscience applications.

## Setting up the Virtual Environment
Before running MRINE, please first create a virtual environment with Python version >= 3.8 and then install the requirements in requirements.txt. Then, navigate to the site-packages directory of the library and create a file named 'mrine.pth', and copy-paste the path to the repository into that file (e.g. 'path-to-repo/MRINE'). 

## Data and Config
The data used for the analyses in the paper are provided in data folder, i.e., data/lorenz for the stochastic Lorenz simulations and data/nhp for the NHP dataset. 
The config files for both single-scale networks and MRINE are also provided in configs folder. 

## Running MRINE on Data for Reproduction
To run MRINE on stochastic Lorenz simulations and the NHP dataset, please run run_kfold_mrine_lorenz.py and run_kfold_mrine_nhp.py scripts, respectively. 
The arguments and their definitions are provided in the scripts. The scripts load the default configs by default. To see the definitions of hyperparameters, 
please see model/config.py.

