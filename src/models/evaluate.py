from sklearn.metrics import accuracy_score, classification_report


def evaluate_model(model, X_val, y_val):
    """
    Evaluate the model on a validation dataset and output accuracy and classification reports.
    """
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    report_dict = classification_report(y_val, preds, output_dict=True)
    report_str = classification_report(y_val, preds)
    
    return acc, report_dict, report_str
