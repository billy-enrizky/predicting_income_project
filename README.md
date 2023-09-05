# Predicting Income Project

This project focuses on predicting income levels using a dataset containing various features. We follow a series of data preprocessing, feature engineering, and model building steps to achieve accurate income predictions.

## 1. Introduction to the Dataset

We begin by loading the dataset and understanding its characteristics:

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('income.csv')

# Display summary information for each column
for col in df.columns:
    counts = df[col].value_counts()
    print(f'Dataframe[{col}]')
    print(counts)
    print('\n')
```

**Dataset Description:**
The dataset contains information on various features, including age, education, occupation, marital status, and more. The target variable is "income," which can be either ">50K" or "<=50K."

We decide to drop the "fnlwgt" column as it is unrelated to income predictions.

## 2. One-Hot Encoding

### One-Hot Encoding Multi-Class Columns

We perform one-hot encoding on columns with multiple classes, such as education, marital status, occupation, race, and native country. This transforms categorical data into numerical format for machine learning.

### Encoding Binary-Class Columns

We encode binary-class columns, such as "gender" and "income," using binary values (0 or 1).

## 3. Feature Selection - Most Correlated with Income

We aim to select the most relevant features for predicting income. To do this, we identify the top 20% of columns that are most correlated with the income column:

```python
# Filtering Top 20% Most Correlated Columns with the Income Column
income_corr = df.corr()['income'].abs()
sorted_income_corr = income_corr.sort_values()
num_cols_to_drop = int(0.8 * len(df.columns))
cols_to_keep = sorted_income_corr.iloc[num_cols_to_drop:].index
df_most_corr = df[cols_to_keep]
```

We also visualize the correlations using a heatmap to identify the most influential features.

## 4. Building a Machine Learning Model

### Why Use a Random Forest Model?

We opt for a Random Forest model due to its ability to handle both numerical and categorical data, as well as its capability to mitigate overfitting compared to individual decision trees.

### Data Splitting

We split the dataset into training and testing sets:

```python
from sklearn.model_selection import train_test_split

X = df.drop('income', axis=1)
y = df['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### Applying The Random Forest Classifier Model

We use the Random Forest Classifier to predict income levels:

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier().fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
```

The model achieves an accuracy of approximately 84.92%.

### Analyzing The Features Importance

We analyze the importance of different features in the Random Forest model:

```python
feature_imp = dict(zip(clf.feature_names_in_, clf.feature_importances_))
feature_imp = {k: v for k,v in sorted(feature_imp.items(), key = lambda x: x[1], reverse=True)}
feature_imp
```

This information helps identify the most crucial factors for predicting income levels.

## 5. Hyperparameter Tuning

We utilize GridSearchCV to find the optimal parameter values for the Random Forest Classification model, specifically focusing on parameters such as n_estimators, max_depth, min_samples_split, and max_features.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [None, 20, 30, 40],
    'min_samples_split': [5, 8, 11],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Create the RandomForestClassifier
rf_classifier = RandomForestClassifier()

# Create the GridSearchCV object
grid_search = GridSearchCV(rf_classifier, param_grid=param_grid, verbose=10, n_jobs=-1)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)
```

We determine the best parameters and evaluate the model's performance with these settings.

## Conclusion

This project demonstrates the process of predicting income levels using a Random Forest Classifier. It encompasses data preprocessing, feature engineering, model building, and hyperparameter tuning. The information gained from feature importance and correlations can be valuable for decision-makers and analysts, providing insights into the factors influencing income predictions.
