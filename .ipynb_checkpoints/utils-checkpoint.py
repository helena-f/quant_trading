
# print evaluation metrics
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
    import matplotlib.pyplot as plt
    
    # Predict and calculate probabilities
    predictions = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Metrics
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    
    # Print metrics
    print(f"\n{model_name} Evaluation Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    if roc_auc is not None:
        print(f"ROC AUC Score: {roc_auc:.2f}")
    
    # Plot ROC Curve
    if roc_auc is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Receiver Operating Characteristic (ROC) Curve: {model_name}")
        plt.legend(loc="best")
        plt.grid()
        plt.show()

#print("hi")