from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Function to create a model using SMOTE
def prepare_model_smote(df, class_col, cols_to_exclude):
    """
    Prepare a model using Synthetic Minority Oversampling Technique (SMOTE).

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        class_col (str): The name of the target class column.
        cols_to_exclude (list): A list of column names to exclude from the features.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, and y_test.
    """
    cols = df.select_dtypes(include=np.number).columns.tolist()
    X = df[cols]
    X = X[X.columns.difference([class_col])]
    X = X[X.columns.difference(cols_to_exclude)]
    y = df[class_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    sm = SMOTE(random_state=0, sampling_strategy=1.0)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test

# Define the models
def run_model(model, X_train, X_test, y_train, y_test):
    """
    Run a specified ensemble model.

    Args:
        model (str): The name of the ensemble model to run ('random', 'adaboost', 'gradient').
        X_train (pd.DataFrame): The training features.
        X_test (pd.DataFrame): The testing features.
        y_train (pd.Series): The training target variable.
        y_test (pd.Series): The testing target variable.

    Returns:
        tuple: A tuple containing the trained model and predictions.
    """
    if model == 'random':
        randomforest = RandomForestClassifier(max_depth=5)
        randomforest.fit(X_train, y_train)
        y_pred = randomforest.predict(X_test)
        randomforest_roc_auc = roc_auc_score(y_test, randomforest.predict(X_test))
        print(classification_report(y_test, y_pred))
        print("The area under the curve is: %0.2f" % randomforest_roc_auc)
        return randomforest, y_pred
    elif model == 'adaboost':
        adaboost = AdaBoostClassifier(n_estimators=100)
        adaboost.fit(X_train, y_train)
        y_pred = adaboost.predict(X_test)
        adaboost_roc_auc = roc_auc_score(y_test, adaboost.predict(X_test))
        print(classification_report(y_test, y_pred))
        print("The area under the curve is: %0.2f" % adaboost_roc_auc)
        return adaboost, y_pred
    elif model == 'gradient':
        gradientboost = GradientBoostingClassifier()
        gradientboost.fit(X_train, y_train)
        y_pred = gradientboost.predict(X_test)
        gradientboost_roc_auc = roc_auc_score(y_test, gradientboost.predict(X_test))
        print(classification_report(y_test, y_pred))
        print("The area under the curve is: %0.2f" % gradientboost_roc_auc)
        return gradientboost, y_pred
    else:
        print("Invalid model name")

# Example usage:
# X_train, X_test, y_train, y_test = prepare_model_smote(df, class_col='churn', cols_to_exclude=['customer_id', 'phone_no', 'year'])
# model, y_pred = run_model('random', X_train, X_test, y_train, y_test)
