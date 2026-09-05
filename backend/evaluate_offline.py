import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

def run_offline_evaluation(
    model_path="backend/fraud_model.pkl",
    test_data_path="data/X_test.csv",
    test_labels_path="data/y_test.csv",
    output_json_path="docs/evaluation_report.json",
    avg_fraud_value=1000.0,       # Average dollar/rupee value per fraud transaction
    false_positive_cost=200.0     # Cost of customer friction / support per false decline
):
    """
    Performs offline evaluation of the trained fraud model on a held-out test set.
    Generates precision, recall, ROC-AUC, confusion matrix, and net business value metrics.
    """
    print("--- Starting Offline Model Evaluation ---")

    # 1. Load trained model using joblib
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train your model first.")
    
    model = joblib.load(model_path)
    print(f"Loaded model from: {model_path}")

    # 2. Load test dataset
    if not os.path.exists(test_data_path) or not os.path.exists(test_labels_path):
        raise FileNotFoundError(f"Test data files missing. Check paths: {test_data_path}, {test_labels_path}")

    X_test = pd.read_csv(test_data_path)
    y_test = pd.read_csv(test_labels_path).values.ravel()

    # 3. Predict Probabilities & Classifications
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]
    else:
        y_probs = model.decision_function(X_test)

    # Standard decision threshold at 0.5
    y_preds = (y_probs >= 0.5).astype(int)

    # 4. Compute ML Performance Metrics
    roc_auc = roc_auc_score(y_test, y_probs)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_preds, average='binary'
    )
    
    cm = confusion_matrix(y_test, y_preds)
    tn, fp, fn, tp = cm.ravel()
    
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # 5. Compute Business Impact / Cost Modeling
    fraud_prevented_amount = float(tp * avg_fraud_value)
    fp_friction_loss = float(fp * false_positive_cost)
    net_protected_value = float(fraud_prevented_amount - fp_friction_loss)

    # 6. Build Structured Output Payload
    report_data = {
        "dataset_summary": {
            "total_test_samples": int(len(y_test)),
            "actual_frauds": int(tp + fn),
            "actual_legitimate": int(tn + fp)
        },
        "performance_metrics": {
            "roc_auc": round(float(roc_auc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "false_positive_rate": round(float(false_positive_rate), 4)
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "business_impact": {
            "fraud_prevented_amount": fraud_prevented_amount,
            "false_positive_friction_cost": fp_friction_loss,
            "net_protected_value": net_protected_value
        }
    }

    # 7. Save to JSON File
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(report_data, f, indent=4)

    print(f"\n--- Offline Evaluation Complete ---")
    print(f"ROC-AUC:   {report_data['performance_metrics']['roc_auc']}")
    print(f"Precision: {report_data['performance_metrics']['precision']}")
    print(f"Recall:    {report_data['performance_metrics']['recall']}")
    print(f"F1 Score:  {report_data['performance_metrics']['f1_score']}")
    print(f"Net Saved: ${net_protected_value:,.2f}")
    print(f"\nReport saved to: {output_json_path}")

    return report_data

if __name__ == "__main__":
    run_offline_evaluation()