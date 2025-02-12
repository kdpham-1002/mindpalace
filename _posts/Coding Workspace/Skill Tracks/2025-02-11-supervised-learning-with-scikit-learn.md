---
layout: post
title: Supervised Learning with Scikit-learn
date: 2025-02-11 18:36 -0800
description: Learn Supervised Learning with scikit-learn for Data Science
author: khoa_pham
categories: [Coding Workspace, Skill Tracks]
tags: [python, interview, coding]
pin: false
math: true
mermaid: true
toc: true
comments: true
---

## Supervised Learning with scikit-learn
* Predicted values are known -> Predict the target values of unseen data, given the features
* EDA (No missing values; Data in numeric format; Stored as Pandas df, NumPy arrays)
* Feature = predictor variable = independent variable
* Target variable = dependent variable = response variable

### 1_Classification
* Target variable consists of categories

#### K_Nearest Neighbors (KNN)
* Looking at `k` closest labeled data points
* Taking a majority vote
* Overfitting and Underfitting
    * Larger k = less complex model = can cause underfitting
    * Smaller k = more complex model = can lead to overfitting

```python
### Fit
# Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier 

X = churn_df[["account_length", "customer_service_calls"]].values
y = churn_df["churn"].values

# Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)


### Predict
X_new = np.array([[30.0, 17.5],
                  [107.0, 24.1],
                  [213.0, 10.9]])

# Predict the labels for the X_new
y_pred = knn.predict(X_new)

# Print the predictions
print("Predictions: {}".format(y_pred)) 



""" Output: 
Predictions: [0 1 0]
# First and third customers will not churn in the new array
"""
```

Train/Test Split and Computing Accuracy:
```python
# Import the module
from sklearn.model_selection import train_test_split

X = churn_df.drop("churn", axis=1).values
y = churn_df["churn"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))



""" Output: 
0.8740629685157422
"""
```

```python
# Create neighbors as a numpy array of values from 1 up to and including 12
neighbors = np.arange(1, 13)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
  
    # Set up a KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=neighbor)
  
    # Fit the model
    knn.fit(X_train, y_train)
  
    # Compute accuracy
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)
print(neighbors, '\n', train_accuracies, '\n', test_accuracies)




""" Output:
[ 1  2  3  4  5  6  7  8  9 10 11 12] 
     {1: 1.0, 2: 0.887943971985993, 3: 0.9069534767383692,..., 11: 0.864432216108054, 12: 0.8604302151075538} 
     {1: 0.7871064467766117, 2: 0.8500749625187406, 3: 0.8425787106446777,..., 11: 0.8598200899550225, 12: 0.8590704647676162}
"""
```

Visualizing Model Complexity:
```python
# Add a title
plt.title("KNN: Varying Number of Neighbors")

# Plot training accuracies
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")

# Plot test accuracies
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")

plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

# Display the plot
plt.show()



""" Output:
Training accuracy decreases and test accuracy increases as the number of neighbors gets larger. 
For the test set, accuracy peaks with 7 neighbors, suggesting it is the optimal value for our model.
"""
```

### 2_Regression
* Target variable is continuous

Creating Features:
```python
import numpy as np

# Create X from the radio column's values
X = sales_df["radio"].values

# Create y from the sales column's values
y = sales_df["sales"].values

# Reshape X
X = X.reshape(-1, 1)

# Check the shape of the features and targets
print(X.shape, y.shape)



""" Output:
 (4546, 1) (4546,)
"""
```

#### Linear Regression
```python
# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create X and y arrays
X = sales_df.drop("sales", axis=1).values
y = sales_df["sales"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X_train, y_train)

# Make predictions
y_pred = reg.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))



""" Output:
Predictions: [53176.66154234 70996.19873235], 
Actual Values: [55261.28 67574.9 ]

# The first two predictions appear to be within around 5% of the actual values from the test set!
"""
```

Regression Performance:
```python
# Import mean_squared_error
from sklearn.metrics import mean_squared_error

# Compute R-squared
r_squared = reg.score(X_test, y_test)

# Compute RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Print the metrics
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))

""" Output:
R^2: 0.9990165886162027
RMSE: 2942.372219812037

# Model explains 99.90% of the variance (almost perfect fit, but could be overfitting.
# On average, predictions are off by about 2942.37 units of the target variable.
"""
```

#### Cross-validation for R-squared
```python
# Import the necessary modules
from sklearn.model_selection import KFold, cross_val_score

# Create a KFold object
kf = KFold(n_splits=6, shuffle=True, random_state=5)

reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = cross_val_score(reg, X, y, cv=kf)

# Print scores
print(cv_scores)

# Print the mean
print(np.mean(cv_results))

# Print the standard deviation
print(np.std(cv_results))

# Print the 95% confidence interval
print(np.quantile(cv_results, [0.025, 0.975]))



""" Output:
[0.74451678 0.77241887 0.76842114 0.7410406  0.75170022 0.74406484]
    0.7536937416666666
    0.012305386274436092
    [0.74141863 0.77191915]

# R-squared for each fold ranged between 0.74 and 0.77. 
# By using cross-validation, we can see how performance varies depending on how the data is split!
# An average score of 0.75 with a low standard deviation is pretty good for a model out of the box.
```

#### Regularized regression: Ridge
```python
# Import Ridge
from sklearn.linear_model import Ridge
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ridge_scores = []
for alpha in alphas:
  
  # Create a Ridge regression model
  ridge = Ridge(alpha=alpha)
  
  # Fit the data
  ridge.fit(X_train, y_train)
  
  # Obtain R-squared
  score = ridge.score(X_test, y_test)
  ridge_scores.append(score)
print(ridge_scores)



""" Output:
[0.9990152104759369, 0.9990152104759373, 0.9990152104759419, 0.9990152104759871, 0.9990152104764387, 0.9990152104809561]

# The scores don't appear to change much as alpha increases, which is indicative of how well the features explain the variance in the target
# Even by heavily penalizing large coefficients, underfitting does not occur.
"""
```

Lasso regression for feature importance:
```python
# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regression model
lasso = Lasso(alpha=0.3)

# Fit the model to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)
plt.bar(sales_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()



""" Ouput:
[ 3.56256962 -0.00397035  0.00496385]

# Expenditure on TV advertising is the most important feature in the dataset to predict sales values.
```

### 3_Fine-tuning your model
* Class imbalance: Uneven frequency of classes
#### Confusion Matrix
	                    | Actual: Positive (1) | Actual: Negative (0)
Predicted: Positive (1)	| True Positive (TP) | False Positive (FP)
Predicted: Negative (0) | False Negative (FN) | True Negative (TN)

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
# Import confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

knn = KNeighborsClassifier(n_neighbors=6)

# Fit the model to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



""" Output:   
    [[116  35]
     [ 46  34]]
                  precision    recall  f1-score   support
    
               0       0.72      0.77      0.74       151
               1       0.49      0.42      0.46        80
    
        accuracy                           0.65       231
       macro avg       0.60      0.60      0.60       231
    weighted avg       0.64      0.65      0.64       231

# The model produced 34 true positives and 35 false positives, meaning precision was less than 50%, which is confirmed in the classification report. # The output also shows a better F1-score for the zero class, which represents individuals who do not have diabetes
"""
```
#### Logistic Regression
```python
# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Instantiate the model
logreg = LogisticRegression()

# Fit the model
logreg.fit(X_train, y_train)

# Predict probabilities
y_pred_probs = logreg.predict_proba(X_test)[:, 1]

print(y_pred_probs[:10])


""" Output:
    [0.26551031 0.18336542 0.12119596 0.15613565 0.49611285 0.44582236
     0.01359235 0.61646125 0.55640546 0.7931187 ]

# Notice how the probability of a diabetes diagnosis for the first 10 individuals in the test set ranges from 0.01 to 0.79.
"""
```

ROC Curve:
```python
# Import roc_curve
from sklearn.metrics import roc_curve

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

plt.plot([0, 1], [0, 1], 'k--')

# Plot tpr against fpr
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Diabetes Prediction')
plt.show()


"""
The ROC curve is above the dotted line, so the model performs better than randomly guessing the class of each observation
"""
```

Area under the ROC curve:
```python
# Import roc_auc_score
from sklearn.metrics import roc_auc_score

# Calculate roc_auc_score
print(roc_auc_score(y_test, y_pred_probs))

# Calculate the confusion matrix
print(confusion_matrix(y_test, y_pred))

# Calculate the classification report
print(classification_report(y_test, y_pred))


""" Output:
    0.8002483443708608
    [[121  30]
     [ 30  50]]
                  precision    recall  f1-score   support
    
               0       0.80      0.80      0.80       151
               1       0.62      0.62      0.62        80
    
        accuracy                           0.74       231
       macro avg       0.71      0.71      0.71       231
    weighted avg       0.74      0.74      0.74       231

# Logistic regression performs better than the KNN model across all the metrics calculated.
# A ROC AUC score of 0.8002 means this model is 60% better than a chance model at correctly predicting labels!
"""
```

#### Hyperparameter tuning 
with GridSearchCV:

```python
# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Set up the parameter grid
param_grid = {"alpha": np.linspace(0.00001, 1, 20)}

# Instantiate lasso_cv
lasso_cv = GridSearchCV(lasso, param_grid, cv=kf)

# Fit to the training data
lasso_cv.fit(X_train, y_train)
print("Tuned lasso paramaters: {}".format(lasso_cv.best_params_))
print("Tuned lasso score: {}".format(lasso_cv.best_score_))



""" Output:
Tuned lasso paramaters: {'alpha': 1e-05}
Tuned lasso score: 0.33078807238121977

# Unfortunately, the best model only has an R-squared score of 0.33, highlighting that using the optimal hyperparameters does not guarantee a high performing model.
"""
```

with RandomizedSearchCV:
```python
# Create the parameter space
params = {"penalty": ["l1", "l2"],
         "tol": np.linspace(0.0001, 1.0, 50),
         "C": np.linspace(0.1, 1.0, 50),
         "class_weight": ["balanced", {0:0.8, 1:0.2}]}

# Instantiate the RandomizedSearchCV object
logreg_cv = RandomizedSearchCV(logreg, params, cv=kf)

# Fit the data to the model
logreg_cv.fit(X_train, y_train)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Best Accuracy Score: {}".format(logreg_cv.best_score_))



""" Ouput:
Tuned Logistic Regression Parameters: {'tol': 0.14294285714285712, 'penalty': 'l2', 'class_weight': 'balanced', 'C': 0.6326530612244898}
Tuned Logistic Regression Best Accuracy Score: 0.7460082633613221

# Even without exhaustively trying every combination of hyperparameters, the model has an accuracy of over 70% on the test set.
"""
```

### 4_Preprocessing and Pipelines
Create dummy variables:
```python
# Create music_dummies
music_dummies = pd.get_dummies(music_df, drop_first=True)
# Print the new DataFrame's shape
print("Shape of music_dummies: {}".format(music_dummies.shape))
```
Ridge Regression model:
```python
# Create X and y
X = music_dummies.drop("popularity", axis=1).values
y = music_dummies["popularity"].values

# Instantiate a ridge model
ridge = Ridge(alpha=0.2)

# Perform cross-validation
scores = cross_val_score(ridge, X, y, cv=kf, scoring="neg_mean_squared_error")

# Calculate RMSE
rmse = np.sqrt(-scores)
print("Average RMSE: {}".format(np.mean(rmse)))
print("Standard Deviation of the target array: {}".format(np.std(y)))



""" Output:
Average RMSE: 8.236853840202299
Standard Deviation of the target array: 14.02156909907019

# An average RMSE of approximately 8.24 is lower than the standard deviation of the target variable (song popularity), suggesting the model is reasonably accurate.
"""
```

Handling missing data:
```python
# Print missing values for each column
print(music_df.isna().sum().sort_values())

# Remove values where less than 5% (<50) are missing
music_df = music_df.dropna(subset=["genre", "popularity", "loudness", "liveness", "tempo"])

# Convert genre to a binary feature
music_df["genre"] = np.where(music_df["genre"] == "Rock", 1, 0)

print(music_df.isna().sum().sort_values())
print("Shape of the `music_df`: {}".format(music_df.shape))
```
Imputation within a pipeline:
```python
# Import modules
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Instantiate an imputer
imputer = SimpleImputer()

# Instantiate a knn model
knn = KNeighborsClassifier(n_neighbors=3)

# Build steps for the pipeline
steps = [("imputer", imputer), 
         ("knn", knn)]

# Create the pipeline
pipeline = Pipeline(steps)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Print the confusion matrix
print(confusion_matrix(y_test, y_pred))



""" Ouput:
    [[79  9]
     [ 4 82]]
"""
```

Centering and scaling for regression:
```python
# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Create pipeline steps
steps = [("scaler", StandardScaler()),
         ("lasso", Lasso(alpha=0.5))]

# Instantiate the pipeline
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)

# Calculate and print R-squared
print(pipeline.score(X_test, y_test))



""" Ouput:
0.6193523316282489

# The model may have only produced an R-squared of 0.619, but without scaling this exact model would have only produced a score of 0.35
"""
```

Centering and scaling for classification:
```python
# Build the steps
steps = [("scaler", StandardScaler()),
         ("logreg", LogisticRegression())]
pipeline = Pipeline(steps)

# Create the parameter space
parameters = {"logreg__C": np.linspace(0.001, 1.0, 20)}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=21)

# Instantiate the grid search object
cv = GridSearchCV(pipeline, param_grid=parameters)

# Fit to the training data
cv.fit(X_train, y_train)
print(cv.best_score_, "\n", cv.best_params_)


""" Output:
0.8425 
 {'logreg__C': 0.1061578947368421}

# Using a pipeline shows that a logistic regression model with "C" set to approximately 0.1 produces a model with 0.8425 accuracy.
"""
```

#### Evaluating multiple models
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
* Regression model performance: 
    * RMSE
    * R-squared
* Classification model performance: 
    * Accuracy
    * Confusion matrix
    * Precision, recall, F1-score
    * ROC AUC

**Note on Scaling:**
* Best to scale our data before evaluating models
* Models affected by scaling:
    * KNN
    * Linear Regression (plus Ridge, Lasso)
    * Logistic Regression
    * Artificial Neural Network

Evaluate regression model performance:
```python
models = {"Linear Regression": LinearRegression(), "Ridge": Ridge(alpha=0.1), "Lasso": Lasso(alpha=0.1)}
results = []

# Loop through the models' values
for model in models.values():
  kf = KFold(n_splits=6, random_state=42, shuffle=True)
  
  # Perform cross-validation
  cv_scores = cross_val_score(model, X_train, y_train, cv=kf)
  
  # Append the results
  results.append(cv_scores)
  
# Create a box plot of the results
plt.boxplot(results, labels=models.keys())
plt.show()


"""
# Lasso regression is not a good model for this problem, while linear regression and ridge perform fairly equally
"""
```

```python
# Import mean_squared_error
from sklearn.metrics import mean_squared_error

for name, model in models.items():
  
  # Fit the model to the training data
  model.fit(X_train_scaled, y_train)
  
  # Make predictions on the test set
  y_pred = model.predict(X_test_scaled)
  
  # Calculate the test_rmse
  test_rmse = mean_squared_error(y_test, y_pred, squared=False)
  print("{} Test Set RMSE: {}".format(name, test_rmse))



""" Output:
Linear Regression Test Set RMSE: 0.11988851505947569
Ridge Test Set RMSE: 0.11987066103299668

# The linear regression model just edges the best performance, although the difference is a RMSE of 0.00001 for popularity
"""
```

Evaluate classification model performance:
```python
# Create models dictionary
models = {"Logistic Regression": LogisticRegression(), "KNN": KNeighborsClassifier(), "Decision Tree Classifier": DecisionTreeClassifier()}
results = []

# Loop through the models' values
for model in models.values():
  
  # Instantiate a KFold object
  kf = KFold(n_splits=6, random_state=12, shuffle=True)
  
  # Perform cross-validation
  cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
  results.append(cv_results)
plt.boxplot(results, labels=models.keys())
plt.show()



"""
# Looks like logistic regression is the best candidate based on the cross-validation results
"""
```

```python
# Create steps
steps = [("imp_mean", SimpleImputer()), 
         ("scaler", StandardScaler()), 
         ("logreg", LogisticRegression())]

# Set up pipeline
pipeline = Pipeline(steps)
params = {"logreg__solver": ["newton-cg", "saga", "lbfgs"],
         "logreg__C": np.linspace(0.001, 1.0, 10)}

# Create the GridSearchCV object
tuning = GridSearchCV(pipeline, param_grid=params)
tuning.fit(X_train, y_train)
y_pred = tuning.predict(X_test)

# Compute and print performance
print("Tuned Logistic Regression Parameters: {}, Accuracy: {}".format(tuning.best_params_, tuning.score(X_test, y_test)))




""" Output:
Tuned Logistic Regression Parameters: {'logreg__C': 0.112, 'logreg__solver': 'newton-cg'}, Accuracy: 0.82

# Built a preprocessing pipeline, and performed hyperparameter tuning to create a model that is 82% accurate in predicting song genres
"""
```
