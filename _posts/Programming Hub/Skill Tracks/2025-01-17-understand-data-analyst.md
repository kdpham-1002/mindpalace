---
layout: post
title: Understand Data Analyst
date: 2025-01-17 09:00 -0800
description: Data Analyst Recap
author: khoa_pham
categories: [Programming Hub, Skill Tracks]
tags: [interview preps, data roles]
pin: false
math: true
mermaid: true
toc: true
comments: true
---

## 1) Exploratory Data Analysis (EDA)
### Data Exploration
#### Checking Data Types & Missing Values
```python
import pandas as pd
books = pd.read_csv("books.csv")
print(books.head())  # View first 5 rows

books.info()  # Summary of dataset
```

#### Descriptive Statistics
```python
# For numerical columns
books.describe() # Summary statistics
```

```python
# For categorical columns
books["genre"].value_counts()
```

#### Data Visualization
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(data=books, x="rating", binwidth=0.1)
plt.show()
```

### Data Cleaning & Imputation

#### Handling Missing Values
```python
# Check for missing values
print(books.isnull().sum())

# Drop missing values
books.dropna(inplace=True)
books.dropna(subset=["rating"], inplace=True)
```

````python
# Impute missing values
books["rating"].fillna(books["rating"].mean(), inplace=True)
books["genre"].fillna(books["genre"].mode()[0], inplace=True)
````

#### Handling Categorical Variables
```python
# Check unique authors
print(books["author"].nunique()) 

# Check for authors starting with "Brown"
books["author"].str.contains("^Brown", case=False)  

# Check for specific genres
books["genre"].str.contains("Thriller|Mystery", case=False) 
```

#### Creating categorical column
```python
books["is_highly_rated"] = books["rating"] > 4.5
```

#### Handling Outliers
```python
# Using IQR method
Q1 = books["rating"].quantile(0.25)
Q3 = books["rating"].quantile(0.75)
IQR = Q3 - Q1

upper = Q3 + 1.5 * IQR
lower = Q1 - 1.5 * IQR
outliers = books[(books["rating"] < lower) | (books["rating"] > upper)]
print(outliers)
```


### Relationships in Data
#### Handling DateTime data
```python
books["year"] = pd.to_datetime(books["year"], format="%Y")
books["year"].dt.year  # Extract year
```

#### Correlation Analysis
```python
books.corr()
sns.heatmap(books.corr(), annot=True)
plt.show()
```

#### KDE Plot
```python
# Update the KDE plot so that marriage duration can't be smoothed too far
sns.kdeplot(data=divorce, x="marriage_duration", hue="num_kids", cut=0)
plt.show()

# Update the KDE plot to show a cumulative distribution function
sns.kdeplot(data=divorce, x="marriage_duration", hue="num_kids", cut=0, cumulative=True)
plt.show()
```


### EDA in action

#### Class imbalance
```python
# Print the relative frequency of Job_Category
print(salaries["Job_Category"].value_counts(normalize=True))
```

#### Cross-tabulation
```python
# Cross-tabulate Job_Category and Company_Size
print(pd.crosstab(salaries["Job_Category"], salaries["Company_Size"],
            values=salaries["Salary_USD"], aggfunc="mean"))
```

#### Generating new features
```python
# Get the month of the response
salaries["month"] = salaries["date_of_response"].dt.month

# Extract the weekday of the response
salaries["weekday"] = salaries["date_of_response"].dt.weekday

# Create a heatmap
sns.heatmap(salaries.corr(), annot=True)
plt.show()

# Find the 25th percentile
twenty_fifth = salaries["Salary_USD"].quantile(0.25)

# Save the median
salaries_median = salaries["Salary_USD"].median()

# Gather the 75th percentile
seventy_fifth = salaries["Salary_USD"].quantile(0.75)
print(twenty_fifth, salaries_median, seventy_fifth)

# Create salary labels
salary_labels = ["entry", "mid", "senior", "exec"]

# Create the salary ranges list
salary_ranges = [0, twenty_fifth, salaries_median, seventy_fifth, salaries["Salary_USD"].max()]

# Create salary_level
# pd.cut() creates bins for salary ranges
salaries["salary_level"] = pd.cut(salaries["Salary_USD"],
                                  bins=salary_ranges,
                                  labels=salary_labels)

# Plot the count of salary levels at companies of different sizes
sns.countplot(data=salaries, x="Company_Size", hue="salary_level")
plt.show()
```
