---
layout: post
title: Conda Environment
date: 2025-01-05 20:30 -0800
description: Set up Conda environments
author: khoa_pham
categories: [Programming Hub, Tips & Tricks]
tags: [python, coding, settings]
pin: false
math: true
mermaid: true
toc: true
comments: true
---

## Anaconda/Conda

[How to Install Anaconda on Mac - Dave Ebbelaar](https://youtu.be/RFeIn2ywxG4?si=nVfrjVQzsGtiu5Sg)  
[Get Started wit Anaconda - Anaconda Cloud Freelearning](https://freelearning.anaconda.cloud/get-started-with-anaconda/18200)

> Learn about [Python environment](https://khoapham1002.github.io/mindpalace/posts/python-environment/)

### Environments Management

```bash
# Display general Conda information
conda info

# Update Conda to the latest version
conda update -n base conda

# List all environments
conda info --envs
# --- or ---
conda env list
```

```bash
# Check the current Python version
python --version

# Update Python within Conda
conda update python

# Install a specific Python version
conda install python=3.12.7
```

```bash
# List revisions of the current environment
conda list --revisions

# Restore environment to a specific revision
conda install --revision=1
```

```bash
# Export all dependencies to a .yml file
conda env export > environment.yml

# Create a new environment from a .yml file
conda env create -f environment.yml
# --- example.yml ---
name: my_project_env
channels:
  - conda-forge
dependencies:
  - python=3.9
  - numpy=1.24.3
  - pandas=1.5.3
  - scikit-learn=1.2.2
```

#### Create and Activate a New Environment

```bash
# Create a new environment
conda create --name conda_env

# Create a new environment with a specific Python version
conda create -n my_project_env python=3.12.7

# Rename an environment
conda rename -n conda_env my_env

# Activate an environment
conda activate conda_env

# Deactivate the current environment
conda deactivate

# Remove an environment
conda env remove --name conda_env
```

#### Kernel and Jupyter Setup

```bash
# Install the IPython kernel package in the environment
conda install ipykernel

# Add the environment as a Jupyter kernel
python -m ipykernel install --user --name=conda_env

# Start Jupyter Notebook
jupyter notebook

# Start Jupyter Lab
jupyter lab

# --- Shutdown Jupyter server (use Ctrl + C) ---
```

### Package Management

```bash
# Install the Anaconda distribution (heavy)
conda install anaconda

# Install Jupyter and its dependencies
conda install jupyter jupyterlab notebook

# Install common data science libraries
conda install scikit-learn numpy pandas matplotlib

# Install a package in a specific environment
conda install -n conda_env seaborn
```

```bash
# Add a channel (e.g., conda-forge)
conda config --add channels conda-forge

# Install a package from a specific channel
conda install -c conda-forge yfinance scikit-learn
conda install -c anaconda pandas-datareader
```

```bash
# List all packages in the current environment
conda list

# Update all packages in the current environment
conda update --all

# Remove all packages in the current environment
conda remove --all
```
