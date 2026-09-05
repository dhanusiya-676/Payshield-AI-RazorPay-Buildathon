import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Generate synthetic transaction data
np.random.seed(42)
num_samples = 1000

amount = np.random.uniform(10, 10000, num_samples)
transaction_velocity = np.random.randint(1, 10, num_samples)
new_device = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])
new_location = np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2])

# Simple fraud labeling rule
is_fraud = (
    (amount > 5000) & (transaction_velocity > 5) |
    (new_device == 1) & (new_location == 1) & (amount > 3000)
).astype(int)

X = pd.DataFrame({
    "amount": amount,
    "transaction_velocity": transaction_velocity,
    "new_device": new_device,
    "new_location": new_location
})
y = pd.Series(is_fraud, name="is_fraud")

# Train / Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Export Test datasets required by evaluate_offline.py
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

# Export Full / Train datasets
X_train.to_csv("data/X_train.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)

print("Data generation complete! Saved files to data/ folder:")
print(" - data/X_test.csv")
print(" - data/y_test.csv")
