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
**Class imbalance:** Uneven frequency of classes

#### Confusion Matrix

| 	                   | Predicted: Positive (1) | Predicted: Negative (0) |
| Actual: Positive (1) | True Positive (TP)      | False Negative (FN)     |
| Actual: Negative (0) | False Positive (FP)     | True Negative (TN)      |

**Accuracy** = (TP + TN) / (TP + TN + FP + FN)   
* Correct predictions out of all predictions
* Best for for balanced datasets

**Precision** = TP / (TP + FP)    
* How many of the predicted positives are actually positive
* High precision means fewer false positives
* Useful when false positives are costly (e.g., detecting diseases, fraud detection)

**Recall** = TP / (TP + FN)
* How many actual positives are correctly identified
* High recall means fewer false negatives
* Useful when false negatives are costly (e.g., cancer detection)

**F1 Score** = 2 * (Precision * Recall) / (Precision + Recall)
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
* 1 = perfect model; 0.5 = random guessing

```python
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred_probs))
```

#### Hyperparameter tuning 
Hyperparameters are model settings chosen before training that affect performance (e.g., k in KNN, alpha in Ridge/Lasso)

**GridSearchCV:** Tests all combinations (best for small parameter sets)

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
param_grid = {"alpha": np.arange(0.0001, 1, 10)}
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid, cv=5)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)
```

**RandomizedSearchCV:** Randomly selects parameter combinations (faster for large sets)

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

**One-Hot Encoding (Dummy Variables):** creates binary columns for each category (e.g., “Male” → [1,0], “Female” → [0,1])

```python
pd.get_dummies(df['category'], drop_first=True)
```

**Scikit-learn OneHotEncoder:** Converts text categories to integers

```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoder.fit_transform(df[['category']])
```

#### Handling Missing Data
**Drop missing values:** Remove rows with missing data

```python
df.dropna(subset=["column_name"])
```

**Imputation:** Fill missing values
* Numeric data: Replace with mean/median.
* Categorical data: Replace with most frequent value.

```python
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="mean")  # Can also use "median" or "most_frequent"
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)
```

#### Scaling: Standardization & Normalization
Features with different scales (e.g., income vs. age) can cause models to behave poorly. For example, KNN, Logistic Regression, and Neural Networks require scaled data.

**Standardization:** Centers data at 0 with unit variance.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Normalization:** Scales data to a fixed range (e.g., 0 to 1)

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### Building Pipelines
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
