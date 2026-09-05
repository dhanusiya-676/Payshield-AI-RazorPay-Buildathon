import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Load Generated Data
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()

# 2. Train Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight="balanced"  # Helps boost recall for minority fraud class
)
model.fit(X_train, y_train)

# 3. Save Model to Backend
os.makedirs("backend", exist_ok=True)
joblib.dump(model, "backend/fraud_model.pkl")

print("Model training complete. Saved to backend/fraud_model.pkl")