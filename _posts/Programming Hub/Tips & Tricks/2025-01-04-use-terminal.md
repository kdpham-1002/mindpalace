---
layout: post
title: Use Terminal
date: 2025-01-04 06:00 -0800
description: Examples of Terminal/Bash commands
author: khoa_pham
categories: [Programming Hub, Tips & Tricks]
tags: [settings, coding]
pin: false
math: true
mermaid: true
toc: true
comments: true
---

> After learning some basic terminal commands, you can/should learn Git!  
> Go to [Use Git](https://khoapham1002.github.io/mindpalace/posts/use-git/).

### Navigate Directories

```bash
# List all files in the current directory
ls

# View contents of the current directory
ls .

# List all files with detailed info (permissions, size, etc.)
ls -la

# List files with human-readable sizes
ls -lh

# View file permissions
ls -l

# List all files in a directory
ls -R [path]
```

```bash
# Display the current working directory
pwd

# Change directory
cd path/to/directory

# Move up to the parent directory
cd ..

# Return to the home directory
cd ~

# Navigate to the previous directory
cd -
```

### Create and Remove Files/Directories

```bash
# Create an empty file
touch filename

# Open a file for editing (default editor)
open filename

# Delete a file
rm filename

# Delete multiple files
rm file1 file2 file3

# Force delete without confirmation
rm -f filename
```

```bash
# Create a new directory
mkdir new_directory

# Create multiple nested directories
mkdir -p dir1/dir2/dir3

# Delete a directory (non-empty)
rm -R directory_name

# Force delete a directory and its contents
rm -rf directory_name
```

### Aliases and Shortcuts

```bash
# Create an alias (temporary for the session)
alias listall="ls -la"

# View existing aliases
alias

# Remove an alias
unalias listall
```

```bash
# Open shell configuration file .zshrc or .bashrc
nano ~/.zshrc

# Add a permanent alias
alias listall="ls -la"
source ~/.zshrc
```

### Shortcuts and Commands

```bash
# Clear the terminal screen
clear

# Shortcut for clearing (Mac)
Cmd + K
```

```bash
# Search your command history
Ctrl + R

# Show all previously executed commands
history

# Run a command from history by its number
!123
```

### Systems Related

```bash
# Display system info
uname -a

# View macOS version
sw_vers

# Display system info
uname -a

# View macOS version
sw_vers
```

```bash
# View file permissions
ls -l

# Make a file executable
chmod +x filename

# Set specific permissions (e.g., read, write, execute)
chmod 755 filename

# Change file owner
sudo chown username filename

# Change owner and group
sudo chown username:groupname filename

# View system information
uname -a
```

```bash
# Check your IP address
ifconfig

# Ping a website or server
ping example.com

# View active network connections
netstat -an

# Test connection to a server (e.g., port 80)
telnet example.com 80
```

```bash
# List all processes
ps -A

# Show running processes with CPU and memory usage
top

# Kill a process by its ID
kill PID

# Force kill a process
kill -9 PID
```