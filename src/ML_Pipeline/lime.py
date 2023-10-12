import lime
import lime.lime_tabular

# Define a function for LIME (Local Interpretable Model-agnostic Explanations)
def lime_explanation(model, X_train, X_test, class_names, chosen_index):
    # Create a LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, 
                                                      feature_names=X_train.columns,
                                                      class_names=class_names, 
                                                      kernel_width=5)
    
    # Choose the specific instance to explain
    chosen_instance = X_test.loc[[chosen_index]].values[0]
    
    # Generate the LIME explanation for the chosen instance
    explanation = explainer.explain_instance(chosen_instance, 
                                            lambda x: model.predict_proba(x).astype(float), 
                                            num_features=10)
    
    # Convert the explanation to a Pyplot figure
    fig = explanation.as_pyplot_figure()
    
    return fig
