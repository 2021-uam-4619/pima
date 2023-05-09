# pima

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
diabetes_df=pd.read_csv('diabetes.csv')
diabetes_df.head()
diabetes_df.describe()
diabetes_df.info
diabetes_df.isnull().sum()
# Check the data types of each column
print(diabetes_df.dtypes)
# Check the class distribution
print(diabetes_df['Outcome'].value_counts())
# Identify duplicate rows
duplicate_rows = diabetes_df.duplicated()
# Count the number of duplicate rows
num_duplicates = duplicate_rows.sum()
print("Number of duplicate rows:", num_duplicates)
!pip install sweetviz
dd=pd.read_csv('diabetes.csv')
import sweetviz as sd
d=sd.analyze(dd)
print(d.show_html())
plt.figure(figsize=(8, 6))
sns.histplot(diabetes_df['Age'], kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.show()
plt.figure(figsize=(8, 6))
sns.boxplot(x='Outcome', y='Glucose', data=diabetes_df)
plt.title('Glucose Levels by Outcome')
plt.xlabel('Outcome')
plt.ylabel('Glucose')
plt.show()
plt.figure(figsize=(8, 6))
sns.barplot(x='Outcome', y='Pregnancies', data=diabetes_df)
plt.title('Number of Pregnancies by Outcome')
plt.xlabel('Outcome')
plt.ylabel('Number of Pregnancies')
plt.show()
plt.figure(figsize=(10, 8))
sns.heatmap(diabetes_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
# Split the dataset into features (X) and target variable (y)
X = diabetes_df.drop('Outcome', axis=1) #seprate independent or dependent features
y = diabetes_df['Outcome']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=25)
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Apply the machine learning algorithm (Logistic Regression)
model = LogisticRegression(max_iter=100)
model.fit(X_train_scaled, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test_scaled)
# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy*100)
# Split the dataset into features (X) and target variable (y)
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Apply the machine learning algorithm (Decision Tree Classifier)
model = DecisionTreeClassifier()
model.fit(X_train_scaled, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test_scaled)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report_str)
# Using GridSearchCv and StandardScalar with Regression Algorithm 
# Split the dataset into features (X) and target variable (y)
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Define the parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}

# Perform grid search
grid_search = GridSearchCV(LogisticRegression(max_iter=100), param_grid)
grid_search.fit(X_train_scaled, y_train)
# Best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy in percentage :", accuracy*100)
# Using RobustScalar and GridSearchCV with DecisionTree Classifier
# Split the dataset into features (X) and target variable (y)
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=25)
# Scale the features using RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Define the parameter grid for hyperparameter tuning
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Perform grid search with cross-validation
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
# Best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)
# Use the best model for prediction
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# KNN Neighbour Algorithm
# Split the dataset into features (X) and target variable (y)
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=35)
# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Apply the KNN algorithm
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test_scaled)
# Split the dataset into features (X) and target variable (y)
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Apply the SVM algorithm
model = SVC(kernel='rbf', C=1.0)
model.fit(X_train_scaled, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test_scaled)
# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# # Naive Bayes algorithm
from sklearn.naive_bayes import GaussianNB
# Split the dataset into features (X) and target variable (y)
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Apply the Naive Bayes algorithm
model = GaussianNB()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)



