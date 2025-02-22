---
layout: post
title: Data Science for Business
date: 2025-02-22 05:47 -0800
description: MGMT 657 Course Notes
author: khoa_pham
categories: [Programming Hub, Course Notes]
tags: [interview preps, data roles]
pin: false
math: true
mermaid: true
toc: true
comments: true
---

## Data Science Methods

**Supervised Learning** (Labeled data):
* **Classification (Predicting Categorical Values):** Logistic Regression, Decision Trees, Random Forests, k-NN, Support Vector Machines (SVM), Gradient Boosting (XGBoost, LightGBM, CatBoost), Neural Networks
* **Regression (Predicting Continuous Values):** Linear Regression, Decision Trees, Random Forests, Ridge Regression, Lasso Regression, Gradient Boosting (XGBoost, LightGBM, CatBoost), Neural Networks, k-NN *(rarely used)*

**Unsupervised Learning** (Unlabeled data):
* **Clustering (Grouping Similar Data):** k-Means, Hierarchical Clustering, DBSCAN
* **Similarity Matching (Finding Closest Matches):** Cosine Similarity, Euclidean Distance
* **Co-occurrence Grouping (Finding Items Bought Together):** Association Rule Mining (Apriori Algorithm)
* **Dimensionality Reduction (Simplifying Data):** Principal Component Analysis (PCA), t-SNE

### 1) Classical Statistics vs. Predictive Modeling
Q1: True or False: Determining the statistical significance of a linear regression model is an objective of classical statistics, but not of predictive modeling.

1. Classical Statistics: Focus on Explanation & Inference   
	In classical statistics, regression models are used to understand relationships between variables and test hypotheses.     
	**Suppose you build a linear regression model:** Testing the Effect of Advertising on Sales
	$$ \text{Sales} = \beta_0 + \beta_1 (\text{Advertising Spend}) + \epsilon $$

	A key goal in classical statistics is to determine whether Advertising Spend significantly affects Sales. You do this by:   
	* Checking the p-value for β₁ (Advertising Spend).    
	* If p-value < 0.05, we reject the null hypothesis (meaning the variable is statistically significant).    
	* If p-value > 0.05, Advertising Spend is not a significant predictor of Sales.    

	✅ Key Focus: Understanding which variables matter and whether they have a statistically significant effect.

2. Predictive Modeling: Focus on Accuracy & Performance   
    In predictive modeling, the primary goal is making accurate predictions rather than interpreting relationships.   
    **Suppose you build a predictive model using multiple features:** Predicting House Prices   
    $$ \text{House Price} = f(\text{Size}, \text{Location}, \text{Bedrooms}, \text{Bathrooms}, \text{Year Built}) $$

    In predictive modeling, we are less concerned about whether each individual predictor is statistically significant. Instead, we evaluate the model’s overall performance using:    
    * R² (coefficient of determination)   
    * Mean Squared Error (MSE)   
    * Root Mean Squared Error (RMSE)   
    * Cross-validation scores   

    Even if some variables have insignificant p-values, they might still improve the model’s predictive power. For example:   
    * Removing an “insignificant” variable might reduce accuracy if it still contributes useful patterns.    
    * Techniques like regularization (Lasso, Ridge Regression) are used to improve prediction quality rather than testing statistical significance.    

    ✅ Key Focus: Minimizing prediction error, not testing significance.



### 2) Causal Modeling   
Q2: What user interface makes customers more or less likely to purchase our products?

Regression   
**Causal Modeling** ✅    
Similarity Matching   
Profiling   


* Regression can model correlations but does not necessarily establish causation.
* Similarity Matching identifies similar customers but does not determine UI impact on purchasing.
* Profiling describes customer behaviors but does not infer causal effects.

The question seeks to understand the cause-and-effect relationship between different user interface (UI) designs and customer purchase behavior.   
**Causal modeling** helps determine whether changes in the UI directly influence the likelihood of a purchase.    
Example Approach:     
* A/B Testing: Show different UI designs to different user groups and compare purchase rates.   
* Instrumental Variables: Control for confounding factors that might affect both UI exposure and purchase decisions.   
* Difference-in-Differences (DiD): Compare purchase behaviors before and after a UI change across treatment and control groups.   



### 3) Classification   
Q3: Which customers are most likely to switch wireless providers?   

Profiling   
Co-occurrence Grouping   
Clustering   
**Classification or class probability estimation** ✅

* Profiling: Describes characteristics of customers who have already churned but does not predict future churners.
* Co-occurrence Grouping: Identifies items frequently purchased together but does not predict churn.
* Clustering: Groups customers based on similarities but does not classify or estimate churn probability directly. However, it can be a preprocessing step to segment customers before applying classification.

The goal is to predict which customers are likely to switch wireless providers, making this a supervised learning problem with a clearly defined target variable (churn or no churn).    
**Classification** models estimate the probability of churn based on customer attributes and behaviors.    
Example Approach:     
* Logistic Regression or Decision Trees to classify customers into “likely to churn” vs. “not likely to churn.”
* Random Forests or Gradient Boosting Models (GBM) for improved predictive accuracy.
* Neural Networks for complex patterns in large datasets.



### 4) Clustering   
Q4: Which of the following business questions is best suited to unsupervised analytics techniques?

Use a machine-learning algorithm to predict house prices based on square footage, location etc. -> Regression   
Determine whether or not a customer will make a purchase after viewing an advertisement. -> Classification   
**Identify segments of the customer population based on: gender, location, age, education, income bracket, and so on.** -> Clustering ✅  
Predict the sentiment of a customer based on a tweet or product review. -> Classification

This question involves grouping customers based on their characteristics without predefined labels, making it a clustering problem, a key technique in unsupervised learning. The goal is to find natural groupings in the data that can help with targeted marketing, personalization, or customer behavior analysis.

* K-Means Clustering – Group customers into distinct segments based on demographic attributes.
* Hierarchical Clustering – Build a tree-like hierarchy of customer groups.
* DBSCAN – Detect clusters with varying densities, identifying core customer groups.



### 5) Hypothesis Testing
Q5: Are store profits significantly higher during the month of December than during the rest of the year?

Clustering   
Database querying   
Predictive Modeling   
**Hypothesis Testing** ✅

* Clustering: Groups data based on similarities but does not test significance.
* Database Querying: Retrieves data but does not perform statistical analysis.
* Predictive Modeling: Forecasts future profits but does not determine significance of past trends.

The question asks whether store profits in December are significantly higher than the rest of the year, which requires statistical inference to compare two groups (December vs. other months).     
**Hypothesis testing** helps determine if the observed difference in profits is statistically significant or due to random variation.   
Example Approach:
* Two-sample t-test: Compare average profits in December vs. other months.
* ANOVA (if comparing multiple months): Analyze variance across all months.
* p-value & confidence intervals: Determine statistical significance.




## Supervised Segmentation

### 1) Entropy