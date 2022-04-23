![GitHub repo size](https://img.shields.io/github/repo-size/chiaraalbi46/EvasiveMovements) [![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)

This repo contains the work made for the project of ***Machine Learning*** course, at the **University of Florence**.

A simple and personal implementation of **Vision Transformer** [link](https://arxiv.org/abs/2010.11929) is provided here.  

<p align="center">
  <img src="./imgs/architecture.png" />
</p>

# Environment and packages

In the following paragraph some steps to recreate a usable environment are explained. Conda package manager and Python 3.8 have been used. 

- A usable conda environment can be directly created from the requirements.txt file, using the command:
    
    ``` git conda create --name <env> --file requirements.txt ```

    The requirements.txt file has been exported from an environment on Windows OS, so probably some packages don't correctly fit with different OS. A new conda environment can of course be created, with these commands:

    ```
    conda create --name ViT python=3.8
    conda activate ViT
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    conda install -c anaconda -c conda-forge -c comet_ml comet_ml 
    ```
    Pytorch have been used as the machine learning framework. We suggest to look to this [link](https://pytorch.org/get-started/locally/) for different settings. CometML support is present [Comet.ml](https://www.comet.ml/site/), to monitor the training and validation metrics. A registration is needed (you can use the Github account). There are [many ways](https://www.comet.ml/docs/python-sdk/advanced/#python-configuration) to integrate comet support in a project, but we suggest to save the API key generated, as described [here](https://www.comet.ml/docs/quick-start/), in a .comet.config file and to copy the following lines in your script:
    ```
    import comet_ml
    experiment = comet_ml.Experiment()
    ```
    The .comet.config file has to be placed in the folder, where the script is located. In the repo a blank .comet.config file is provided.

# Download CIFAR-10 and CIFAR-100
Using Windows 10 OS we encountered some difficulties to download CIFARs dataset through the simple function of torchvision. We solved the problem executing two additional lines only the first time we downloaded the dataset. Please look at **download_cifar.py** to get the datasets. 

# Experiments
In order to make experiments on Vision Transformer performances, execute **main.py** for training from scratch or pretraining, **fine_tune.py** for fine tuning on a pretrained model. Look inside the files to get more information on the hyper-parameters. 

# "Inspecting ViT"
Interesting visualizations of some parts of the model are provided, to better understand what happens during the training process. In particular we have implemented:
- Linear embedding weights visualization
- Position embeddings similarities visualization

in **embedding_weights_plot.py** and

- Heads visualization (this was not in the original paper)
- Attention rollout visualization

in **attention_plots.py**