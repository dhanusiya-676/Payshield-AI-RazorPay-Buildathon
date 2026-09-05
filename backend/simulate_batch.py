import pandas as pd
import requests

# Load synthetic or test transactions and send batch requests to FastAPI
df = pd.read_csv("data/X_test.csv").head(100) # Run batch simulation for 100 samples
results = []

for _, row in df.iterrows():
    payload = row.to_dict()
    res = requests.post("http://127.0.0.1:8000/predict-risk", json=payload)
    results.append(res.json())

print(f"Successfully processed {len(results)} transactions in batch mode.")