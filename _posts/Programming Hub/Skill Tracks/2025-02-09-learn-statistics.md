---
layout: post
title: Learn Statistics
date: 2025-02-11 18:14 -0800
description: Statistics Fundamentals with Python
author: khoa_pham
categories: [Programming Hub, Skill Tracks]
tags: [python, interview preps, coding]
pin: false
math: true
mermaid: true
toc: true
comments: true
---

## 1. Intro to Statistics

### Summary Statistics
* Descriptive and Inferential Statistics
    * Numeric (Quantitative) Data: Continuous (measured) vs Discrete (counted)
    * Categorical (Qualitative) Data: Nominal (unordered) vs Ordinal (ordered)
* Measure of Center
    * Mean vs Median vs Mode
    * When skewed, use **median** because it’s resistant to outliers
* Measure of Spread
    * IQR, Variance, Standard Deviation, Mean Absolute Deviation, Quantiles
* **All in One** -> .describe()

### Probability and Distributions
* Independent -> Sampling with replacement
    * E.g: flipping a coin 3 times, rolling a die twice
* Dependent -> Sampling without replacement
    * E.g: randomly selecting 5 products from the assembly line to test for quality assurance

* Discrete Uniform Distributions
    * E.g: the outcome of rolling a die
    * Law of Large Numbers
        * E.g: If coin is flipped 1000 times, p is getting closer to 0.5
* Continuous Uniform Distribution
    * E.g: the time it takes to wait for a bus
    * P(4 < wait ≤ 7) = P(wait ≤ 7) - P(wait ≤ 4)
        * uniform.cdf(7, 0, 12) - uniform.cdf(4, 0, 12)
    * Genereate 10 random wait times of between 0 to 5min
        * uniform.rvs(0, 5, size=10)

* Binomial Distribution
    * E.g: the number of heads when flipping a coin 10 times
    * Don’t work with dependent trials!
    * Flip 3 coins with 50% chance of success 10 times
        * binom.rvs(3, 0.5, size=10)
    * binom.pmf(num heads, num trials, prob of heads)
        * binom.pmf(7, 10, 0.5)
    * Probability of more than 7 heads
        * 1 - binom.cdf(7, 10, 0.5)

### More Distributions and CLT
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
    * E.g: number of products sold each week
    * Lambda (λ) is the distribution’s peak
    * In terms of rate (Poisson): λ = 8 adoptions per week
    * Avg # of adoptions per week is 8, P(adt = 5) = ?
        * poisson.pmf(5, 8)
    * Avg # of adoptions per week is 8, P(adt <= 5) = ?
        * poisson.cdf(5, 8)
* Exponential Distribution
    * E.g: amount of time until the nextcustomer makes a purchase
    * On average, one customer service ticket is created every 2 minutes
    * λ = 0.5 customer service tickets created each minute
    * In terms of time between events (exponential): 1/λ = 1/0.5 = 2
        * P (wait > 4 min) = 1 - expon.cdf(4, scale=2)
        * P (1 min < wait < 4 min) = expon.cdf(4, scale=2) - expon.cdf(1, scale=2)
* (Student's) t-distribution
* Log-normal distribution

### Correlation and Experimental Design
* sns.scatterplot(), sns.lmplot(), .corr()
* Pearson, Kendall, Spearman Correlation
* When rely on linear regression and it’s highly skewed, use transformations:
    * log transformation -> np.log()
    * square root transformation
    * reciprocal transformation
* Confounding variables
    * Holidays (x) -> Retail Sales (y)
    * Special Deals (confounder)


## 2. Intro to Regression 

### Linear Regression
Predict a continuous numerical value based on one or more input features. It finds a straight-line relationship between the input and output.      
For example:
* Predicting sales revenue based on advertising spend.
* Estimating a person’s weight based on height.
* Forecasting temperature based on time of day.

| House Size (sq ft) | Price ($) |
|--------------------|-----------|
| 1000               | 200,000   |
| 1500               | 250,000   |
| 2000               | 300,000   |
| 2500               | 350,000   |

**Linear Regression equation:**   

$$ 
\text{Price} = \beta_0 + \beta_1 \times \text{Size} + \epsilon
$$

where:
* β₀ (Intercept): The predicted price when the size is 0 (not meaningful here)
* β₁ (Slope): How much price increases for every extra square foot
* ε (Error Term): The difference between actual and predicted price

> If β₁ = 100, then a 1000 sq ft house → $200,000

#### R-squared (R^2)
Measures how much of the variation in the target variable is explained by the model.

* `R^2 = 1` → Overfitting (the model is too specific to training data and won’t work well on new data).
* `R^2 = 0` → The model is as good as a random guess.

* If `R^2 = 0.90` → 90% of house price changes can be explained just by knowing the house size. The remaining 10% might be due to other factors like location, age, etc.
* If  `R^2 = 0.30` → The model explains only `30%`, meaning it’s not very reliable.

#### Adjusted R-squared
Penalizes adding more features to the model. It’s useful when comparing models with different numbers of features.
* Use Adjusted R^2 to check for overfitting, since adding useless variables can inflate R^2 but lower Adjusted R^2 

* `Adjusted R^2 = 1` → The model is perfect.
* `Adjusted R^2 = 0` → The model is as good as a random guess.

#### Residual Standard Error (RSE)
Measures the average difference between actual and predicted values.
* Use RSE when comparing models with different predictors, because it tells us if adding variables improves predictions.
* Use RMSE for general error interpretation, since it is easier to understand.

#### Mean Squared Error (MSE)
Measures the average squared difference between actual and predicted values.
* Comparing models with different predictors

#### Root Mean Squared Error (RMSE)
Measures the average error in the same unit as the target variable 
* Used when large errors are more problematic (e.g., House prices, stock market, medical cost predictions).
    * A $20,000 error increases RMSE much more than a $5,000 error.

#### Mean Absolute Error (MAE)
Takes the absolute value of errors
* Used when all errors are equally bad (e.g., Delivery times, customer waiting times, forecasting demand).
    * A 2-minute delay is just as annoying as a 10-minute delay


#### Models Comparison

| Model | RSE (Lower = Better) | RMSE (Lower = Better) | MAE (Lower = Better) | R^2 Score (Higher = Better) | Adjusted R^2 (Higher = Better) |
|-------|----------------------|-----------------------|----------------------|-----------------------------|--------------------------------|
| Model 1 (House Size Only)    | $29,687 | $29,687 | $23,837 | 0.685 | 0.682 |
| Model 2 (House Size + Bedrooms + Location) | $21,655 | $21,655 | $17,243 | 0.836 | 0.831 |
| Model 3 (House Size + Bedrooms + Location + Useless Features) | $21,835 | $21,835 | $17,203 | 0.837 | 0.828 |

**📌 Explanation of the Models**   
1. Model 1 (Only House Size)   
    * R^2 = 0.685  → The model explains 68.5% of the variance in house prices.   
    * RSE = $29,687$ → On average, predictions deviate by $29,687$ from actual prices.
    * MAE = $23,837$ → The model is off by about $23,837$ on average.

    ❌ Model 1 is too simple:   
    * It has much higher RSE and MAE, meaning predictions are less reliable.

2. Model 2 (House Size + Bedrooms + Location)
    * R^2 = 0.836  (higher) → Now the model explains 83.6% of the variance, meaning it is much better than Model 1.
    * RSE drops to $21,655$ (better) → Predictions are now more accurate.
    * MAE drops to $17,243$ (better) → The average error is much lower.

    ✅ Model 2 is the best model because:   
    * It has the lowest RSE ($21,655$).
    * It has the highest Adjusted  R^2  (0.831).
    * It balances complexity vs. accuracy well.

3. Model 3 (Adding Useless Features: Street Name + Paint Color)
    * R^2  only slightly increases to 0.837, but…
    * RSE slightly increases to $21,835$ → Predictions become slightly less accurate.
    * Adjusted  R^2  drops to 0.828 (bad sign!), meaning the extra features are not actually helping the model.

    ❌ Model 3 is overfitting:   
    *  R^2  increased slightly, but Adjusted  R^2  dropped.
    * RSE slightly worsened despite extra variables.
    * Conclusion: The extra features do not improve prediction accuracy.



### Logistic Regression
Predict a categorical (binary) outcome, such as “Yes/No”. It estimates the probability of an event happening.   
For example:
* Email spam detection (Spam vs. Not Spam).
* Disease prediction (Has Disease vs. No Disease).
* Fraud detection (Fraudulent Transaction vs. Normal Transaction).

| Credit Score | Approved? (1=Yes, 0=No) |
|--------------|-------------------------|
| 400          | 0 (No)                  |
| 550          | 0 (No)                  |
| 650          | 1 (Yes)                 |
| 700          | 1 (Yes)                 |

**Logistic Regression equation:**   

$$
P(\text{Approved}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 \times \text{Credit Score})}}
$$ 

where:
* β₀ (Intercept) and β₁ (Coefficient) control the decision boundary.
* The output is a probability between 0 and 1.
* If P(Approved) > 0.5, predict Yes, otherwise predict No.

> If β₀ = -5, β₁ = 0.01, then a credit score of 700 → P(Approved) ≈ 0.8 (Likely Approved)

#### Confusion Matrix
A table that describes the performance of a classification model. It shows the number of True Positives, True Negatives, False Positives, and False Negatives.

| Actual/Predicted | Predicted ✅ | Predicted ❌ |
|------------------|-------------|--------------|
| Actual ✅         | TP (Correct)| FN (Missed Case) |
| Actual ❌         | FP (False Alarm) | TN (Correct) |

* True Positive (TP): Predicted Yes, and the actual outcome is Yes.
* True Negative (TN): Predicted No, and the actual outcome is No.
* False Positive (FP): Predicted Yes, but the actual outcome is No (Type I Error).
* False Negative (FN): Predicted No, but the actual outcome is Yes (Type II Error).

#### Accuracy
Measures the proportion of correct predictions.   
`Accuracy = (TP + TN) / (TP + TN + FP + FN)`
* High accuracy doesn’t always mean a good model. It can be misleading if the classes are imbalanced.

#### Precision
Measures the proportion of true positive predictions among all positive predictions (e.g, the proportion of actual spam emails among all emails predicted as spam).   
`Precision = TP / (TP + FP)`
* High precision means fewer false positives.
* Use Precision when False Positives are bad (jail sentencing, fraud detection (wrong fraud alerts), medical trials, autonomous vehicles).

| Suspect | Actual Innocence/Guilt | Predicted by Model  | Outcome                |
|---------|-------------------------|--------------------|------------------------|
| A       | ❌ (Not Guilty)         | ✅ (Wrongly Convicted) | 🚨 False Positive   |
| B       | ❌ (Not Guilty)         | ❌ (Correct)        |                        |
| C       | ✅ (Guilty)             | ✅ (Correct)        |                        |
| D       | ✅ (Guilty)             | ❌ (Wrongly Released) | 🚨 False Negative    |

* A False Positive (FP) means an innocent person is sent to jail (BAD!).
* A False Negative (FN) means a guilty person is wrongly released (still bad, but not as bad as FP in this case).

#### Recall (Sensitivity)
Measures the proportion of true positive predictions among all actual positive instances (e.g, the proportion of actual spam emails among all emails that are actually spam).   
`Recall = TP / (TP + FN)`
* High recall means fewer false negatives.
* Use Recall when False Negatives are bad (e.g., cancer detection (missing a patient), medical diagnoses, fire alarms, security threats).

| Patient | Actual Cancer Status | Predicted by Model | Outcome                |
|---------|----------------------|--------------------|------------------------|
| A       | ✅ (Positive)        | ❌ (Not Detected)  | 🚨 False Negative      |
| B       | ✅ (Positive)        | ✅ (Detected)      |                        |
| C       | ❌ (Negative)        | ✅ (Wrongly Diagnosed) | 🚨 False Positive   |
| D       | ❌ (Negative)        | ❌ (Correct)       |                        |

* A False Positive (FP) means a healthy person is wrongly told they have cancer (bad, but they will take more tests to confirm).
* A False Negative (FN) means a cancer patient is wrongly told they are healthy (VERY BAD – could lead to death).

#### F1 Score
It balances precision and recall, especially when classes are imbalanced.    
`F1 Score = 2 * (Precision * Recall) / (Precision + Recall)`
* Use F1 Score when both False Positives and False Negatives are bad (e.g, fraud detection, spam filtering, rare diseases).

#### ROC Curve
Receiver Operating Characteristic (ROC) curve is a graphical representation of the model’s performance at various thresholds.
* It plots the True Positive Rate (Recall) against the False Positive Rate (1 - Specificity).
* The Area Under the Curve (AUC) measures the model’s ability to distinguish between classes.
    * AUC = 1 → Perfect model
    * AUC = 0.5 → Random guessing

#### Models Comparison

| Model                                      | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------------------------------------------|----------|-----------|--------|----------|---------|
| Model 1 (Credit Score Only)                | 0.620    | 0.622     | 0.647  | 0.635    | 0.653   |
| Model 2 (Credit Score + Income + Loan Amount) | 0.940    | 0.924     | 0.961  | 0.942    | 0.973   |
| Model 3 (Adding Useless Features: ZIP Code + Marital Status) | 0.935    | 0.924     | 0.951  | 0.937    | 0.972   |

**📌 Explanation of the Models**   

1. Model 1 (Only Credit Score)
    * Accuracy = 62% → Predicts correctly about 62% of the time.
    * Precision = 62% → When it predicts loan approval, 62% of the time it’s correct.
    * Recall = 64.7% → Captures 64.7% of actual approved loans.
    * ROC-AUC = 0.653 → Not great at distinguishing approvals from rejections.

    ❌ Problem?
    * Credit score alone doesn’t fully determine loan approval.   
    * Conclusion: Doesn't capture enough information to make accurate predictions.

2. Model 2 (Credit Score + Income + Loan Amount)
    * Accuracy = 94% → Predicts correctly 94% of the time.
    * Precision = 92% → When predicting approval, it’s correct 92% of the time.
    * Recall = 96% → Captures 96% of actual approvals.
    * F1 Score = 94% → A balance of precision & recall.
    * ROC-AUC = 0.973 → Almost perfect model for distinguishing approvals.

    ✅ Why is this better?
    * Adding income and loan amount helps correctly predict approvals and denials.
    * Higher accuracy & recall = Fewer false positives & false negatives.   
    * Conclusion: This is the best model, as it balances precision, recall, and AUC.   

3. Model 3 (Adding ZIP Code + Marital Status – Useless Features)
    * Accuracy drops slightly to 93.5% (no improvement).
    * ROC-AUC = 0.972 (almost identical to Model 2).
    * Adjusted Precision/Recall but no major change.

    ❌ What happened?
    * Adding ZIP code and marital status didn’t help.
    * Model overfits by learning useless patterns.
    * No real improvement over Model 2.    
    * Conclusion: Adding unnecessary features does not improve performance and can make the model more complex without benefits.
