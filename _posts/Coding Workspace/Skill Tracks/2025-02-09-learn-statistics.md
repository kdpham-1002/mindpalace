---
layout: post
title: Learn Statistics
date: 2025-02-11 18:14 -0800
description: Statistics Fundamentals with Python
author: khoa_pham
categories: [Coding Workspace, Skill Tracks]
tags: [python, interview preps, coding]
pin: false
math: true
mermaid: true
toc: true
comments: true
---

## 1. Intro to Statistics in Python
### 1_Summary Statistics
* Descriptive and Inferential Statistics
    * Numeric (Quantitative) Data: Continuous (measured) vs Discrete (counted)
    * Categorical (Qualitative) Data: Nominal (unordered) vs Ordinal (ordered)
* Measure of Center
    * Mean vs Median vs Mode
    * .agg([np.mean, np.median, statistics.mode])
    * When skewed, use **median**
* Measure of Spread
    * Variance, Standard Deviation, Mean Absolute Deviation, Quantiles
    * np.var(), np.std()
        - `(, ddof = 1)` -> calculates sample  
    * np.quantile(, np.linspace(start, stop, num)), plt.boxplot()
    * `from scipy.stats import iqr` -> iqr()
    * **All in One** -> .describe()

### 2_Probability and Distributions
* Independent -> Sampling with replacement
* Dependent -> Sampling without replacement
    * np.random.seed(123), .sample()
* Discrete Distributions, Law of Large Numbers
    * If coin is flipped 1000 times, p is getting closer to 0.5
* Continuous Uniform Distribution
    * P(4 < wait ≤ 7) = P(wait ≤ 7) - P(wait ≤ 4)
        * uniform.cdf(7, 0, 12) - uniform.cdf(4, 0, 12)
    * Genereate 10 random wait times of between 0 to 5min
        * uniform.rvs(0, 5, size=10)
* Binomial Distribution
    * Don’t work with dependent trials!
    * Flip 3 coins with 50% chance of success 10 times
        * binom.rvs(3, 0.5, size=10)
    * binom.pmf(num heads, num trials, prob of heads)
        * binom.pmf(7, 10, 0.5)
    * Probability of more than 7 heads
        * 1 - binom.cdf(7, 10, 0.5)

### 3_More Distributions and CLT
* (Standard) Normal Distribution
    * Percent of women are 154 - 157cm (mean = 161, SD = 7)
        * norm.cdf(157, 161, 7) - norm.cdf(154, 161, 7)
    * What height are 90% of women taller than?
        * norm.ppf((1-0.9), 161, 7)
    * Generate 10 random heights
        * norm.rvs(161, 7, size=10)
* Central Limit Theorem
    * Sampling distribution becomes closer to the normal distribution as n increases
* Poisson Distribution
    * Lambda (λ) is the distribution’s peak
    * In terms of rate (Poisson): λ = 8 adoptions per week
    * Avg # of adoptions per week is 8, P(adt = 5) = ?
        * poisson.pmf(5, 8)
    * Avg # of adoptions per week is 8, P(adt <= 5) = ?
        * poisson.cdf(5, 8)
* Exponential Distribution
    * On average, one customer service ticket is created every 2 minutes
    * λ = 0.5 customer service tickets created each minute
    * In terms of time between events (exponential): 1/λ = 1/0.5 = 2
        * P (wait > 4 min) = 1- expon.cdf(4, scale=2)
        * P (1 min < wait < 4 min) = expon.cdf(4, scale=2) - expon.cdf(1, scale=2)
* (Student's) t-distribution
* Log-normal distribution

### 4_Correlation and Experimental Design
* sns.scatterplot(), sns.lmplot(), .corr()
* Pearson, Kendall, Spearman Correlation
* When rely on linear regression and it’s highly skewed, use transformations:
    * log transformation -> np.log()
    * square root transformation
    * reciprocal transformation
* Confounding variables
    * Holidays (x) -> Retail Sales (y)
    * Special Deals (confounder)
