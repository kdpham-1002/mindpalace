---
layout: post
title: Python Environments
date: 2025-01-04 20:00 -0800
description: Set up Python environments
author: khoa_pham
categories: [Coding Workspace, Setups]
tags: [python, venv]
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