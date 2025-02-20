---
layout: post
title: Learn Machine Learning
date: 2025-02-11 18:36 -0800
description: Intro to Machine Learning
author: khoa_pham
categories: [Programming Hub, Skill Tracks]
tags: [python, interview preps, coding]
pin: false
math: true
mermaid: true
toc: true
comments: true
---

## Introduction to Machine Learning

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




## Supervised Learning with scikit-learn   
Predicted values are known -> Predict the target values of unseen data, given the features
* Feature = predictor variable = independent variable
* Target variable = dependent variable = response variable

### Classification
Predicting categories from labeled data (e.g., spam or not spam)
* Steps in Classification:
    1.	Train a model using labeled data.
    2.	Model learns patterns from the data.
    3.	Pass unseen data to the model.
    4.	Model predicts class labels.

#### K_Nearest Neighbors (KNN)
Predicts labels based on k closest neighbors (majority vote).

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```

Tuning k (overfitting vs underfitting):
* Smaller k = more complex model = can lead to overfitting
* Larger k = less complex model = can cause underfitting

#### Model Evaluation
$$ 
Accuracy = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
$$

Train/Test Split:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))  # Accuracy
```

Plot Model Complexity:
```python
plt.plot(neighbors, train_accuracies.values(), label="Train Accuracy")
plt.plot(neighbors, test_accuracies.values(), label="Test Accuracy")
```


### Regression
Predicting continuous values from labeled data (e.g., house prices, blood glucose levels, stock prices)

#### Linear Regression
Equation:  $$ y = ax + b $$
* a (slope), b (intercept) are model parameters.

Loss Function: Minimize Residual Sum of Squares (RSS).

#### Model Evaluation
1. R-squared: Explained variance of the model
    * 1 = perfect fit, 0 = no fit

```python
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(reg.score(X_test, y_test))  # R-squared
```

2. Mean Squared Error (MSE): Average squared difference between predicted and actual values

3. Root Mean Squared Error (RMSE): Average error in the model's predictions
    * Same units as target variable
    * Lower RMSE = better model

```python
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred, squared=False))  # RMSE
```


#### Cross-validation
Splitting data once may not reflect true model performance -> Prevent overfitting.       
* Perform k-fold cross-validation:   
	1.	Split data into k equal parts.
	2.	Train on (k-1) parts and test on 1 part.
	3.	Repeat for all k parts.

```python
from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(reg, X, y, cv=kf)
print(scores.mean())  # Average performance
```

#### Regularized Regression
Standard regression can overfit when features have large coefficients. Penalizes large coefficients to reduce overfitting.

1. Ridge Regression: L2 regularization
Penalizes large coefficients to reduce overfitting.

$$ \text{Loss Function} = \text{RSS} + \alpha \sum a^2 $$

```python
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
```

2. Lasso Regression: L1 regularization
Performs feature selection by shrinking some coefficients to zero.

$$ \text{Loss Function} = \text{RSS} + \alpha \sum |a| $$

```python
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
plt.bar(feature_names, lasso.coef_)
plt.show()
```

### Fine-tuning your model
* Class imbalance: Uneven frequency of classes
#### Confusion Matrix

| 	                   | Predicted: Positive (1) | Predicted: Negative (0) |
| Actual: Positive (1) | True Positive (TP)      | False Negative (FN)     |
| Actual: Negative (0) | False Positive (FP)     | True Negative (TN)      |

* **Accuracy** = (TP + TN) / (TP + TN + FP + FN)
    * Correct predictions out of all predictions
    * Best for for balanced datasets
* **Precision** = TP / (TP + FP) 
    * How many of the predicted positives are actually positive
    * High precision means fewer false positives
    * Useful when false positives are costly (e.g., detecting diseases, fraud detection)
* **Recall** = TP / (TP + FN)
    * How many actual positives are correctly identified
    * High recall means fewer false negatives
    * Useful when false negatives are costly (e.g., cancer detection)
* **F1 Score** = 2 * (Precision * Recall) / (Precision + Recall)
    * High F1 score means a good balance between precision and recall.
    * Useful when both false positives and false negatives are costly
    * Best for imbalanced datasets

```python
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```


#### Logistic Regression & ROC Curve
Logistic Regression outputs probabilities.
* If  p > 0.5  → Predicted 1, otherwise 0.

ROC Curve (Receiver Operating Characteristic):
* Plots True Positive Rate (Recall) vs. False Positive Rate.
* Helps find the best classification threshold.

Plot ROC Curve:
```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
```

Area under the ROC curve (ROC AUC):
Measures model performance across all classification thresholds
* 1 = perfect model, 0.5 = random guessing

```python
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred_probs))
```

#### Hyperparameter tuning 
Hyperparameters are model settings chosen before training that affect performance (e.g., k in KNN, alpha in Ridge/Lasso)

* GridSearchCV: Tests all combinations (best for small parameter sets)

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
param_grid = {"alpha": np.arange(0.0001, 1, 10)}
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid, cv=5)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)
```

* RandomizedSearchCV: Randomly selects parameter combinations (faster for large sets)

```python
from sklearn.model_selection import RandomizedSearchCV
ridge_cv = RandomizedSearchCV(ridge, param_grid, cv=5, n_iter=2)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)
```


### Preprocessing and Pipelines

#### Handling Categorical Features
Machine learning models require numeric data. Some features are text-based (e.g., “Male”, “Female”)   
Convert categorical data using:

* One-Hot Encoding (Dummy Variables): creates binary columns for each category (e.g., “Male” → [1,0], “Female” → [0,1])

```python
pd.get_dummies(df['category'], drop_first=True)
```

* Scikit-learn OneHotEncoder: Converts text categories to integers

```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoder.fit_transform(df[['category']])
```

#### Handling Missing Data
* Drop missing values: Remove rows with missing data

```python
df.dropna(subset=["column_name"])
```

* Imputation: Fill missing values
    * Numeric data: Replace with mean/median.
	* Categorical data: Replace with most frequent value.

```python
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="mean")  # Can also use "median" or "most_frequent"
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)
```

#### Standardization & Normalization
Features with different scales (e.g., income vs. age) can cause models to behave poorly. For example, KNN, Logistic Regression, and Neural Networks require scaled data.

* Standardization: Centers data at 0 with unit variance.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

* Normalization: Scales data to a fixed range (e.g., 0 to 1)

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### Pipelines
* Pipelines automate preprocessing & modeling.   
* Prevents data leakage (ensures transformations apply to test data properly).

```python
# Example Pipeline: Imputation + Scaling + Model
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])
pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)
```

```python
# Example Pipeline: Scaling + KNN
from sklearn.pipeline import Pipeline
steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=6))]
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)
```

```python
# Example Pipeline: GridSearchCV with Pipeline
from sklearn.model_selection import GridSearchCV
param_grid = {"knn__n_neighbors": np.arange(1, 50)}
cv = GridSearchCV(pipeline, param_grid=param_grid)
cv.fit(X_train, y_train)
print(cv.best_params_, cv.best_score_)
```


#### Evaluating multiple models

**Model Selection:**

| Criteria          | Example Models                      |
|-------------------|-------------------------------------|
| Small Data        | KNN, Logistic Regression            |
| Fast Training     | Decision Tree, Naive Bayes          |
| High Accuracy     | Random Forest, Neural Networks      |
| Interpretable     | Linear Regression, Decision Tree    |

**Guiding Principles:**
* Size of the dataset
    * Fewer features = simpler model, faster training time
    * Some models require large amounts of data to perform well
* Interpretability
    * Some models are easier to explain, which can be important for stakeholders
    * Linear regression has high interpretability, as we can understand the coefficients
* Flexibility
    * May improve accuracy, by making fewer assumptions about data
    * KNN is a more flexible model, doesn't assume any linear relationships

**Note on Metrics:**
* Regression model performance: RMSE, R-squared
* Classification model performance: Confusion matrix, Accuracy, Precision, recall, F1-score, ROC AUC

**Note on Scaling:**
* Best to scale our data before evaluating models
* Models affected by scaling: KNN, Linear Regression (plus Ridge, Lasso), Logistic Regression, Artificial Neural Network

```python
# Evaluating multiple models
from sklearn.model_selection import cross_val_score, KFold
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}
results = []
for model in models.values():
    kf = KFold(n_splits=5, shuffle=True)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=kf)
    results.append(scores)
```

```python
# Boxplot of model performance
import matplotlib.pyplot as plt
plt.boxplot(results, labels=models.keys())
plt.show()
```

```python
# Evaluating on test data
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    print(f"{name} Test Accuracy: {model.score(X_test_scaled, y_test)}")
```
