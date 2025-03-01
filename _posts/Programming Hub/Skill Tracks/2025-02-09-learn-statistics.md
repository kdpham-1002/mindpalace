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

## 1) Intro to Statistics

### Summary Statistics
Types of Statistics:   
* Descriptive Statistics: Summarizes and describes data (e.g., mean, median, mode).   
* Inferential Statistics: Uses a sample to make predictions about a larger population.   

Types of Data:
1. Numeric (Quantitative):  
    * Continuous: Measured values (e.g., speed, height).
    * Discrete: Counted values (e.g., number of pets).
2. Categorical (Qualitative):  
    * Nominal: No inherent order (e.g., country, gender).
    * Ordinal: Ordered categories (e.g., survey ratings: disagree, neutral, agree).


Measures of Center:
* Mean (Average)
    * The sum of values divided by count.
    * Sensitive to outliers.
    * Normal Distribution? → Use mean.
    * Formula: $$ \text{Mean} = \frac{\sum X}{N} $$
```python
np.mean(msleep['sleep_total'])
```

* Median
    * The middle value when data is sorted.
    * Less sensitive to outliers than the mean.
    * Skewed Data? → Use median.
```python
np.median(msleep['sleep_total'])
```

* Mode
    * The most frequent value in the dataset.
    * Categorical Data? → Use mode.
```python
statistics.mode(msleep['vore'])
```

Measures of Spread:
* Variance
	* Measures how far data points are from the mean.
	* Formula: $$ \sigma^2 = \frac{\sum (X - \bar{X})^2}{N-1} $$
```python
np.var(msleep['sleep_total'], ddof=1)
```

* Standard Deviation (SD)
	* Square root of variance, representing spread in original units.
    * Formula: $$ \sigma = \sqrt{\frac{\sum (X - \bar{X})^2}{N-1}} $$
```python
np.std(msleep['sleep_total'], ddof=1)
```

* Mean Absolute Deviation (MAD)
	* Measures absolute differences from the mean.
	* Difference from SD: MAD treats all deviations equally, while SD gives larger weight to extreme values.
    * Formula: $$ \text{MAD} = \frac{\sum |X - \bar{X}|}{N} $$
```python
np.mean(np.abs(msleep['sleep_total'] - np.mean(msleep['sleep_total'])))
```

* Quantiles & Interquartile Range (IQR)
	* Quantiles: Divide data into equal parts (e.g., median = 50th percentile).
	* IQR: Difference between Q3 (75th percentile) and Q1 (25th percentile).
    * Formula: $$ \text{IQR} = Q3 - Q1 $$
```python
from scipy.stats import iqr
iqr(msleep['sleep_total'])
```

* Outliers Dectection
    * Lower Bound: Q1 - 1.5 * IQR
    * Upper Bound: Q3 + 1.5 * IQR
```python
Q1 = msleep['sleep_total'].quantile(0.25)
Q3 = msleep['sleep_total'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
```

**All in One** -> .describe()
```python
msleep['sleep_total'].describe()
```

### Probability and Distributions
Probability of an event: $$ P(\text{event}) = \frac{\text{# of ways event can happen}}{\text{total # of possible outcomes}} $$
* E.g: Coin flip -> $$  P(\text{Heads}) = \frac{1}{2} = 50\% $$

Independent -> Sampling with replacement
* E.g: flipping a coin 3 times, rolling a die twice

Dependent -> Sampling without replacement
* E.g: randomly selecting 5 products from the assembly line to test for quality assurance

Discrete Uniform Distributions
* Describe probabilities of different possible outcomes.
* Expected Value (Mean of a probability distribution): $$ E(X) = \sum P(X) \times X $$

* E.g: the outcome of rolling a fair die -> $$ E(X) = (1 \times \frac{1}{6}) + (2 \times \frac{1}{6}) + … + (6 \times \frac{1}{6}) = 3.5 $$

Law of Large Numbers
* As the number of trials increases, the sample mean gets closer to the population mean.
* E.g: If coin is flipped 1000 times, p is getting closer to 0.5

Continuous Uniform Distribution
* Equal probability for all values in a range.
* E.g: the time it takes to wait for a bus

```python
from scipy.stats import uniform
# Probability of waiting between 4 and 7 minutes
# P(4 < wait ≤ 7) = P(wait ≤ 7) - P(wait ≤ 4)
uniform.cdf(7, 0, 12) - uniform.cdf(4, 0, 12)

# Generate 10 random wait times between 0 and 12 minutes
uniform.rvs(0, 12, size=10)
```

Binomial Distribution
* Models the number of successes in repeated trials.
* Don’t work with dependent trials!
* Formula: $$ P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k} $$
* Expected Value: $$ E(X) = n \times p $$

```python
from scipy.stats import binom
# Probability of 7 heads in 10 flips
# binom.pmf(num heads, num trials, prob of heads)
binom.pmf(7, 10, 0.5)

# Probability of more than 7 heads
1 - binom.cdf(7, 10, 0.5)

# Flip 3 coins with 50% chance of success 10 times
binom.rvs(3, 0.5, size=10)
```

### More Distributions and CLT
(Standard) Normal Distribution
* A symmetric, bell-shaped curve that represents a probability distribution.
* Total area under the curve = 1, the curve extends infinitely but never reaches 0.
* Empirical Rule:
    * 68% of data falls within 1σ.
    * 95% of data falls within 2σ.
    * 99.7% of data falls within 3σ.

```python
from scipy.stats import norm
# norm.cdf(x, mean, std) → Finds probability P(X ≤ x).
# Probability that a woman’s height is less than 154 cm, given µ=161, σ=7)
norm.cdf(154, 161, 7)

# Percent of women are 154 - 157cm (mean = 161, SD = 7)
norm.cdf(157, 161, 7) - norm.cdf(154, 161, 7)

# norm.ppf(prob, mean, std) → Finds X for given probability.
# What height are 90% of women taller than?
norm.ppf((1-0.9), 161, 7)

# norm.rvs(mean, std, size=n) → Generates random samples.
# Generate 10 random heights
norm.rvs(161, 7, size=10)
```

Central Limit Theorem
* The sampling distribution of the mean approaches a normal distribution as sample size increases, regardless of population distribution.
    * Large enough sample size (n ≥ 30)
    * Random & independent sampling.



Poisson Distribution
* Models the number of events occurring in a fixed interval of time or space.
* E.g: number of products sold each week, customer service requests per hour, website visits per minute
* Lambda (λ) is the distribution’s peak
    * In terms of rate (Poisson): λ = 8 adoptions per week
    * Avg # of adoptions per week is 8, P(adt = 5) = ?
        * poisson.pmf(5, 8)
    * Avg # of adoptions per week is 8, P(adt <= 5) = ?
        * poisson.cdf(5, 8)

```python
from scipy.stats import poisson
# In terms of rate (Poisson): λ = 8 adoptions per week
# Avg # of adoptions per week is 8, P(adt = 5) = ?
poisson.pmf(5, 8)

# Avg # of adoptions per week is 8, P(adt <= 5) = ?
poisson.cdf(5, 8)

# Generate 10 random adoptions per week
poisson.rvs(8, size=10)
```

Exponential Distribution
* Models waiting times between Poisson events.
* E.g: time until the next customer makes a purchase, time between bus arrivals, time between earthquakes

```python
from scipy.stats import expon
# On average, one customer service ticket is created every 2 minutes
# λ = 0.5 customer service tickets created each minute
# In terms of time between events (exponential): 1/λ = 1/0.5 = 2

# P (wait > 4 min) = 
1 - expon.cdf(4, scale=2)
# P (1 min < wait < 4 min) = 
expon.cdf(4, scale=2) - expon.cdf(1, scale=2)
```

(Student's) t-distribution
* Similar to the normal distribution but with heavier tails.
* Used when the sample size is small or the population standard deviation is unknown.
* Lower df → thicker tails, higher df → closer to normal distribution.


Log-normal distribution
* The logarithm of the data follows a normal distribution.
* E.g: stock prices, income, sales, population of cities, blood pressure


### Correlation and Experimental Design

Correlation
* Measures the strength and direction of a linear relationship between two variables.
* Pearson, Kendall, Spearman Correlation

```python
df['X'].corr(df['Y'])
```

* When rely on linear regression and it’s highly skewed, use transformations:
    * log transformation -> np.log()
    * square root transformation -> np.sqrt()
    * reciprocal transformation -> 1 / x

Confounding variables
* Holidays (x) -> Retail Sales (y)
* Special Deals (confounder)




## 2) Regression with statsmodels
### Intro to Regression
#### Linear Regression
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

##### R-squared (R^2)
Measures how much of the variation in the target variable is explained by the model.

* `R^2 = 1` → Overfitting (the model is too specific to training data and won’t work well on new data).
* `R^2 = 0` → The model is as good as a random guess.

* If `R^2 = 0.90` → 90% of house price changes can be explained just by knowing the house size. The remaining 10% might be due to other factors like location, age, etc.
* If  `R^2 = 0.30` → The model explains only `30%`, meaning it’s not very reliable.

##### Adjusted R-squared
Penalizes adding more features to the model. It’s useful when comparing models with different numbers of features.
* Use Adjusted R^2 to check for overfitting, since adding useless variables can inflate R^2 but lower Adjusted R^2 

* `Adjusted R^2 = 1` → The model is perfect.
* `Adjusted R^2 = 0` → The model is as good as a random guess.

##### Residual Standard Error (RSE)
Measures the average difference between actual and predicted values.
* Use RSE when comparing models with different predictors, because it tells us if adding variables improves predictions.
* Use RMSE for general error interpretation, since it is easier to understand.

##### Mean Squared Error (MSE)
Measures the average squared difference between actual and predicted values.
* Comparing models with different predictors

##### Root Mean Squared Error (RMSE)
Measures the average error in the same unit as the target variable 
* Used when large errors are more problematic (e.g., House prices, stock market, medical cost predictions).
    * A $20,000 error increases RMSE much more than a $5,000 error.

##### Mean Absolute Error (MAE)
Takes the absolute value of errors
* Used when all errors are equally bad (e.g., Delivery times, customer waiting times, forecasting demand).
    * A 2-minute delay is just as annoying as a 10-minute delay


##### Models Comparison

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



#### Logistic Regression
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

##### Confusion Matrix
A table that describes the performance of a classification model. It shows the number of True Positives, True Negatives, False Positives, and False Negatives.

| Actual/Predicted | Predicted ✅ | Predicted ❌ |
|------------------|-------------|--------------|
| Actual ✅         | TP (Correct)| FN (Missed Case) |
| Actual ❌         | FP (False Alarm) | TN (Correct) |

* True Positive (TP): Predicted Yes, and the actual outcome is Yes.
* True Negative (TN): Predicted No, and the actual outcome is No.
* False Positive (FP): Predicted Yes, but the actual outcome is No (Type I Error).
* False Negative (FN): Predicted No, but the actual outcome is Yes (Type II Error).

##### Accuracy
Measures the proportion of correct predictions.   
`Accuracy = (TP + TN) / (TP + TN + FP + FN)`
* High accuracy doesn’t always mean a good model. It can be misleading if the classes are imbalanced.

##### Precision
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

##### Recall (Sensitivity)
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

##### F1 Score
It balances precision and recall, especially when classes are imbalanced.    
`F1 Score = 2 * (Precision * Recall) / (Precision + Recall)`
* Use F1 Score when both False Positives and False Negatives are bad (e.g, fraud detection, spam filtering, rare diseases).

##### ROC Curve
Receiver Operating Characteristic (ROC) curve is a graphical representation of the model’s performance at various thresholds.
* It plots the True Positive Rate (Recall) against the False Positive Rate (1 - Specificity).
* The Area Under the Curve (AUC) measures the model’s ability to distinguish between classes.
    * AUC = 1 → Perfect model
    * AUC = 0.5 → Random guessing

##### Models Comparison

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


### Simple Linear Regression Modeling
#### What is Regression?
* Explores the relationship between a response (dependent) variable and one or more explanatory (independent) variables.
* Used for prediction and understanding relationships.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("swedish_motor_insurance.csv")

# Scatter plot with trend line
sns.regplot(x="n_claims", y="total_payment_sek", data=df, ci=None)
plt.show()
```

> n_claims (explanatory) predicts total_payment_sek (response).    
> A positive correlation suggests more claims lead to higher total payments.   

#### Fitting a Linear Model

Equation of a Simple Linear Regression Model:   
$$ Y = \text{Intercept} + (\text{Slope} \times X) $$

```python
from statsmodels.formula.api import ols

# Fit model
model = ols("total_payment_sek ~ n_claims", data=df).fit()
print(model.params)

# Output:
# Intercept    19.99
# n_claims      3.41
```

> Intercept (19.99): The base payment when there are zero claims.   
> Slope (3.41): For each additional claim, total payment increases by 3.41 SEK.


### Predictions and Model Objects
#### Making Predictions
```python
import numpy as np

# New data to predict
explanatory_data = pd.DataFrame({"n_claims": np.arange(10, 101, 10)})

# Predict using model
prediction_data = explanatory_data.assign(
    total_payment=model.predict(explanatory_data)
)

print(prediction_data)
```

> The model predicts total payments based on the number of claims.

#### Checking Model Attributes
```python
print(model.fittedvalues)  # Predictions on the original dataset
print(model.resid)         # Residuals (errors)
print(model.summary())     # Full model report
```

Interpreting Model Output:   
> R-squared → Measures how well the model explains the variance.   
> P-value → Checks if the explanatory variable significantly predicts the response.


### Assessing Model Fit
#### Coefficient of Determination (R-squared)
```python
print(model.rsquared)

# Output: 0.64
```

> 64% of the variance in total_payment_sek is explained by n_claims.   
> Ranges from 0 to 1; Higher R² → Better model fit.

#### Residual Standard Error (RSE)
```python
mse = model.mse_resid
rse = np.sqrt(mse)
print("RSE:", rse)

# Output: RSE: 74.15
```

> RSE indicates the average distance between observed and predicted values.   
> Lower RSE → Better model predictions. 

#### Root Mean Squared Error (RMSE)
```python
rmse = np.sqrt(np.mean(model.resid**2))
print("RMSE:", rmse)

# Output: RMSE: 72.01
```

> RMSE indicates the average error in predictions.   
> Lower RMSE → Better model predictions.

#### Residual Diagnostics
```python
import statsmodels.api as sm

# Residual plot
sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.show()

# Q-Q Plot (normality check)
sm.qqplot(model.resid, line="45")
plt.show()
```

> Random scatter in residuals plot → Good model.   
> Straight line in Q-Q plot → Residuals are normally distributed.

### Simple Logistic Regression Modeling
#### Why Logistic Regression?
* Used when the response variable is binary (e.g., Churn: 1 = Yes, 0 = No).    
* Predicts probabilities instead of continuous values.

#### Fitting a Logistic Model
```python
from statsmodels.formula.api import logit

# Fit logistic model
log_model = logit("has_churned ~ time_since_last_purchase", data=churn).fit()
print(log_model.params)

# Output:
# Intercept                   -0.035
# time_since_last_purchase     0.269
```

> Intercept: Log-odds of churn when time since last purchase is zero.   
> Slope: For each additional day since the last purchase, the log-odds of churn increases by 0.269.
>> Positive coefficient (0.269) → Higher recency increases churn probability.

#### Making Predictions
```python
explanatory_data = pd.DataFrame({"time_since_last_purchase": np.arange(-1, 6.25, 0.25)})

prediction_data = explanatory_data.assign(
    has_churned = log_model.predict(explanatory_data)
)
print(prediction_data)
```
> The model predicts the probability of churn based on the time since the last purchase.

#### Odds Ratio
```python
prediction_data["odds_ratio"] = prediction_data["has_churned"] / (1 - prediction_data["has_churned"])
print(prediction_data)
```
> Odds Ratio: Probability of churn vs. not churn.   
> Odds Ratio > 1 → Higher probability of churn.

#### Model Evaluation
##### Confusion Matrix
```python
import numpy as np

actual = churn["has_churned"]
predicted = np.round(log_model.predict())

conf_matrix = pd.crosstab(actual, predicted)
print(conf_matrix)
```

| Actual \ Predicted | No Churn (0) | Churn (1) |
|--------------------|--------------|-----------|
| No Churn (0)       | TN = 141     | FP = 59   |
| Churn (1)          | FN = 111     | TP = 89   |

* True Positives (TP): Correctly predicted churn cases.
* False Positives (FP): Predicted churn but no churn occurred.
* False Negatives (FN): Missed actual churn cases.
* True Negatives (TN): Correctly predicted no churn cases.

##### Accuracy

$$ \frac{\text{TP} + \text{TN}}{\text{Total Predictions}} $$

```python
accuracy = (conf_matrix.iloc[0, 0] + conf_matrix.iloc[1, 1]) / conf_matrix.values.sum()
print("Accuracy:", accuracy)

# Output: Accuracy: 0.55
```
> Accuracy: 55% of predictions were correct.

##### Sensitivity (Recall)

$$ \frac{\text{TP}}{\text{TP} + \text{FN}} $$

```python
sensitivity = conf_matrix.iloc[1, 1] / (conf_matrix.iloc[1, 0] + conf_matrix.iloc[1, 1])
print("Sensitivity:", sensitivity)

# Output: Sensitivity: 0.44
```

> Sensitivity: 44% of actual churn cases were correctly predicted.   
> Higher sensitivity → Fewer missed churn cases.

##### Specificity

$$ \frac{\text{TN}}{\text{TN} + \text{FP}} $$

```python
specificity = conf_matrix.iloc[0, 0] / (conf_matrix.iloc[0, 0] + conf_matrix.iloc[0, 1])
print("Specificity:", specificity)

# Output: Specificity: 0.70
```

> Specificity: 70% of actual non-churn cases were correctly predicted.   
> Higher specificity → Fewer false alarms.