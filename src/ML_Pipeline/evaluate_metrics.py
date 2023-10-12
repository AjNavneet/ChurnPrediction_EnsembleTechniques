# Function to create a confusion matrix
def confusion_matrix(y_test, y_pred):
    # Import necessary library
    from sklearn.metrics import confusion_matrix
    
    # Generate the confusion matrix
    confusion_matrix_result = confusion_matrix(y_test, y_pred)
    print(confusion_matrix_result)
    
    # Extract TP, FP, TN, FN values from the confusion matrix
    tn, fp, fn, tp = confusion_matrix_result.ravel()
    print('TN: %0.2f' % tn)
    print('TP: %0.2f' % tp)
    print('FP: %0.2f' % fp)
    print('FN: %0.2f' % fn)

# Function for creating a ROC curve
def roc_curve(logreg, X_test, y_test):
    # Import necessary libraries
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    
    # Calculate the ROC AUC score
    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    
    # Calculate the ROC curve data points
    fpr, tpr, thresholds = roc_curve(y_test, logreg.predict(X_test))
    
    # Create the ROC curve plot
    plt.figure()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Plot the worst-case line
    plt.plot([0, 1], [0, 1], 'b--')
    
    # Plot the ROC curve of the logistic regression model
    plt.plot(fpr, tpr, color='darkorange', label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    
    # Add labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    # Save the ROC curve plot (uncomment this line to save the plot to a file)
    # plt.savefig('Log_ROC')
    # Display the plot (uncomment this line to display the plot)
    # plt.show()
