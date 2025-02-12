---
layout: post
title: Learn Python Fundamentals
date: 2025-02-08 19:10 -0800
description: Python Fundamentals and Pandas Basics
author: khoa_pham
categories: [Coding Workspace, Skill Tracks]
tags: [python, interview, coding]
pin: false
math: true
mermaid: true
toc: true
comments: true
---

## 1. Intro to Python
### 1_Python basics
* Calculations
* Variables and types (int, float, str, bool)

### 2_Lists
* Subsetting, slicing; list of lists
* Replace, extend, delete
* Make copy of list 

### 3_Functions and Packages
* Familiar functions
    * help(pow), help(max),…
    * type(), len(), int(), str(), sorted(iterable =, reverse =)
* String methods
    * .upper(), .count(), .index()
    * .append(), .remove(), .reverse()
* Import packages

### 4_NumPy
* NumPy Fundamentals

### 5_Matplotlib
* Histogram
* Line Plot
* Scatter Plot
* Customizations (size, color, grid…)

### 6_Dictionaries
* Create, access dictionaries
* Add, delete; dictionary in dictionaries

### 7_Pandas
* pd.read_csv('cars.csv', index_col = 0)
* Pandas Series vs DataFrame; print columns
* Indexing; loc vs iloc

### 8_Logic, Control Flow, Filtering and Loops
* Comparison, Boolean Operators
* If-Elif-Else
* While loop
* For loop
    * Loop using enumerate()
    * Loop over list of lists
    * Loop over dictionary
    * Loop over NumPy array
    * Loop over DataFrame
    * Add column; .apply(str.upper)

## 2. Functions in Python
### 1_User-defined Functions
### 2_Arguments and Scope
### 3_Lambda Functions and Error-handling
## 3. Data Manipulation with Pandas
### 1_Sorting and Subsetting
* .head(), .info(), .shape, .describe()
* .values, .columns, .index
* Sorting, subsetting, adding
    * .sort_values()
    * .isin()


### 2_Aggregating Dataframes
* .mean(), .median(), .max(), .min()
* .agg([iqr, np.median])
* .cumsum(), .cummax()
* .drop_duplicates(subset=["store", "department"])
* .value_counts(sort=True, normalize=True)
    * Adding ‘normalize’ argument to get proportion
* .groupby("type")[["unemployment", "fuel_pricel"]].agg([np.min, np.mean])
* .pivot_table()
    * filling in any missing values with 0 and summing all rows and columns
sales.pivot_table(values="weekly_sales", index="department", columns="type", fill_value=0, margins=True)
### 3_Slicing and Indexing
* Set and reset index
* Subsetting with.loc[]
* Multi-level indexes
* Sorting by index values
* Slicing index values, in both directions, time series, by row/column number
* Subsetting pivot tables and calculating

### 4_Handling and Visualizing Data
* Finding, removing, replacing missing values
* List of dictionaries; Dictionary of lists
* CSV to df; df to CSV
