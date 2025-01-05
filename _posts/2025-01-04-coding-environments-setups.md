---
layout: post
title: Coding Environments Setups
date: 2025-01-04 18:43 -0800
description: Set up Python, Anaconda, VSCode environments
author: khoa_pham
categories: [Coding Workspace, Settings]
tags: [python, anaconda, vscode]
pin: false
math: true
mermaid: true
toc: true
comments: true
---

## Python

### Environment Management

```bash
# Create a virtual environment in the current directory
python3 -m venv .venv

# Create a virtual environment with a specific Python version
python3.9 -m venv my_py39_env
```

```bash
# Activate the environment on macOS/Linux
source my_project_env/bin/activate

# View installed packages in the environment
pip3 list

# Deactivate the active virtual environment
deactivate

# Delete the virtual environment folder (after deactivation)
rm -rf .venv
```

```bash
# Save the current environment's dependencies to a requirements file
pip3 freeze > requirements.txt

# Install all packages listed in a requirements.txt file
pip3 install -r requirements.txt
# --- requirements.txt ---
numpy==1.22.4
pandas==1.4.3
matplotlib>=3.3,<4.0
scikit-learn>=0.24
```

### Package Management

```bash
# Upgrade pip to the latest version
pip3 install --upgrade pip

# Install specific packages
pip3 install jupyter
pip3 install pandas numpy matplotlib

# Uninstall a specific package
pip3 uninstall package_name
```



## Anaconda

[How to Install Anaconda on Mac - Dave Ebbelaar](https://youtu.be/RFeIn2ywxG4?si=nVfrjVQzsGtiu5Sg)
[Get Started wit Anaconda - Anaconda Cloud Freelearning](https://freelearning.anaconda.cloud/get-started-with-anaconda/18200)

### Environment Management

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
conda create --name example

# Create a new environment with a specific Python version
conda create -n my_project_env python=3.12.7

# Rename an environment
conda rename -n example my_env

# Activate an environment
conda activate example

# Deactivate the current environment
conda deactivate

# Remove an environment
conda env remove --name example
```

#### Kernel and Jupyter Setup

```bash
# Install the IPython kernel package in the environment
conda install ipykernel

# Add the environment as a Jupyter kernel
python -m ipykernel install --user --name=example

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
conda install -n example seaborn
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

## VSCode

[The Ultimate VS Code Setup for Data Science & AI](https://doc.clickup.com/9015213037/d/h/8cnjezd-17675/ddd52c673443975?irclickid=Wnz1XKUrGxyKWfFRwl3uy0zbUkCRCQ3RITrTxU0&utm_source=ir&utm_medium=cpc&utm_campaign=ir_cpc_at_nnc_pro_trial_all-devices_cpc_lp_x_all-departments_x_Datalumina%20B.V.&utm_content=&utm_term=1416724&irgwc=1)

### Navigating

- Cmd + 0/1/B                       // Focus on sidebar, editor, hide sidebar
- Cmd + Ctrl + right                // Move to the right editor
- Opt + Cmd + left/rights           // Move between tabs
- Opt + up/down                     // Move lines
- Cmd + F -> "word" -> Opt + Enter      // Find all occurrences

### Multi cursors

- Opt + Cmd + up/down         // Add cursors above/below
- Opt + click                 // Add cursors with mouse
- Cmd + D or Shift + Cmd + L  // Select next occurrence or all occurrences

### Word wrap

- Opt + Z                    // Toggle word wrap
