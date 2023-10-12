# Import the required libraries
import pickle
from ML_Pipeline.utils import read_data, inspection, null_values
from ML_Pipeline.ml_model import prepare_model_smote, run_model
from ML_Pipeline.evaluate_metrics import confusion_matrix, roc_curve
from ML_Pipeline.lime import lime_explanation
import matplotlib.pyplot as plt

# Read the initial dataset
df = read_data("../input/data_regression.csv")

# Perform data inspection and cleaning
x = inspection(df)

# Drop the rows with null values
df = null_values(df)

### Run the random forest model with sklearn ###

## Prepare the dataset (select numerical columns and exclude specified columns)
X_train, X_test, y_train, y_test = prepare_model_smote(df, class_col='churn',
                                                     cols_to_exclude=['customer_id', 'phone_no', 'year'])

# Train the model (Random Forest in this case)
model_rf, y_pred = run_model('random', X_train, X_test, y_train, y_test)  # Change the model name accordingly

## Performance Metrics ##
# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
# print(conf_matrix)

# Plot the ROC curve and save the figure
roc_val = roc_curve(model_rf, X_test, y_test)
plt.savefig("../output/ROC_curves/ROC_Curve_rf.png")  # Save the ROC curve image

## Save the trained model to a pickle file
pickle.dump(model_rf, open('../output/models/model_rf.pkl', 'wb'))

# Generate a LIME explanation report for the model
lime_exp = lime_explanation(model_rf, X_train, X_test, ['Not Churn', 'Churn'], 1)
lime_exp.savefig('../output/LIME_reports/lime_report_rf.jpg')
