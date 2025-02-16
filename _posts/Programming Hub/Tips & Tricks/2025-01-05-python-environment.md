---
layout: post
title: Python Environment
date: 2025-01-05 20:00 -0800
description: Set up Python environments
author: khoa_pham
categories: [Programming Hub, Tips & Tricks]
tags: [python, coding, settings]
pin: false
math: true
mermaid: true
toc: true
comments: true
---

## Python

I prefer using [Conda environment](https://khoapham1002.github.io/mindpalace/posts/conda-environment/) for managing Python packages, but you can also use Python's built-in `venv` module or `pip` for package management.

### Environments Management

```bash
# Create a virtual environment in the current directory
python3 -m venv py_env

# Create a virtual environment with a specific Python version
python3.9 -m venv my_py39_env
```

```bash
# Activate the environment on macOS/Linux
source py_env/bin/activate

# View installed packages in the environment
pip3 list

# Deactivate the active virtual environment
deactivate

# Delete the virtual environment folder (after deactivation)
rm -rf py_env
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