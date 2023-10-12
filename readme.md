# Churn Prediction using Ensemble Techniques

## Business Objective

In our case study, we will be working on a churn dataset. Churned Customers are those who have decided to end their relationship with their existing company.

XYZ is a service-providing company that provides customers with a one-year subscription plan for their product. The company wants to know if the customers will renew the subscription for the coming year or not.

---

## Data Description

This data provides information about a video streaming service company, where they want to predict if the customer will churn or not. The CSV consists of around 2000 rows and 16 columns.

---

## Tech Stack

- Language: Python
- Libraries: NumPy, pandas, matplotlib, scikit-learn, pickle, imbalanced-learn (imblearn), LIME

## Approach

1. Importing the required libraries and reading the dataset.
2. Feature Engineering
    - Dropping unwanted columns.
3. Model Building
    - Performing train-test split
    - Random Forest Model
    - AdaBoost Model
    - Gradient Boosting Model
4. Model Validation (Predictions)
    - Recall score
    - Precision score
    - F1-score
    - ROC and AUC
5. Feature Importance
    - Create a function to find important features.
    - Plot the features.
6. LIME Implementation
    - Define a function for implementing the LIME technique over the dataset.

## Modular Code Overview

1. **input**: Contains all the data for analysis, including a CSV file.
2. **src**: This is the most important folder of the project. It contains all the modularized code for all the above steps in a modularized manner. This folder consists of:
    - `Engine.py`
    - `ML_Pipeline`

The `ML_Pipeline` is a folder that contains all the functions put into different Python files, which are appropriately named. These Python functions are then called inside the `Engine.py` file.

3. **output**: The output folder contains three subfolders:
    - **LIME_reports**: Contains the LIME reports generated for all three algorithms.
    - **Models**: Contains the models generated for all three algorithms.
    - **ROC_curves**: Contains the ROC curves generated for all three algorithms.

4. **lib**: This is a reference folder. It contains the original IPython notebook that we saw in the videos.

---

## Concepts Explored:

1. Introduction to ensemble techniques.
2. Understanding the working of Random Forest, AdaBoost, and Gradient Boosting algorithms.
3. Using Python libraries such as matplotlib for data interpretation and advanced visualizations.
4. Data inspection and cleaning.
5. Using scikit-learn library to build the Random Forest, AdaBoost, and Gradient Boosting models.
6. Splitting the dataset into train and test using scikit-learn.
7. Making predictions using the trained model.
8. Gaining confidence in the model using metrics such as ROC, AUC, recall, precision, and F1 score.
9. Handling unbalanced data using the SMOTE method.
10. Performing feature importance.
11. Evaluating the ROC curve results across multiple models.
12. Evaluating the different models with respect to the feature importance results generated.
13. Understanding the concept of LIME in machine learning.
14. Implementing the LIME technique on the dataset.
