# PAYSHIELD AI ENGINE 🛡️
*Enterprise-Grade Fraud Defense & Risk Intelligence Pipeline*

[![Razorpay Buildathon](https://img.shields.io/badge/Razorpay-Buildathon-blue?style=for-the-badge)](https://razorpay.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)

---

## 🎯 1. Project Overview & Track Selection
* **Track:** **AI Risk Manager** (Stop merchants from losing money to fraud, returns, and chargebacks with measured precision and recall).
* **The Problem:** Modern online payment channels face sophisticated fraud rings and velocity abuse, leading to chargeback penalties and lost revenue. Manual rules are too slow and rigid, while unconstrained machine learning can cause high false-positive rates that hurt conversion.
* **The Solution:** **PAYSHIELD AI ENGINE** combines deterministic risk rules (velocity checks, Geo-IP, device fingerprinting) with a robust Scikit-Learn/Pandas classification pipeline to score transactions in milliseconds via FastAPI, minimizing false-positive costs while blocking high-risk payloads.

---

## 🏗️ 2. System Architecture

The engine is built on a modular multi-tier architecture separating the client console, API gateway, core intelligence services, and the offline model lifecycle layer.

![PayShield Architecture](docs/PayShield.jpg)

### Core Components:
1. **Frontend Dashboard & Operator Console:** Captures real-time transaction payloads and renders interactive UI elements (Risk Score Gauge, Feature Contributions Chart, Offline Evaluation Reports).
2. **Middleware & API Gateway:** Single secure routing point managing endpoint traffic (e.g., `/predict-risk`).
3. **Core Backend & Intelligence Layer:** 
   * **Fraud Detection Service:** FastAPI/Uvicorn asynchronous low-latency transaction scoring.
   * **Risk Engine Core:** Real-time execution of velocity tracking, Geo-IP analysis, and device fingerprinting.
   * **AI/ML Inference Engine:** Pre-trained model (`fraud_model.pkl`) evaluating multi-feature risk probability.
   * **Offline Evaluation Service:** Logs metrics to an offline report database for auditing model performance.
4. **Data & Model Lifecycle Layer:** Reproducible training pipelines (`generate_data.py`, `train_model.py`), dataset versioning (`X_train.csv`, `y_train.csv`), and batch simulation tools.

---

## 📁 Repository Structure

```text
PAYSHIELD-AI-RAZORPAY-BUILDATHON/
├── backend/
│   ├── main.py               # FastAPI application & entry point
│   ├── fraud_model.pkl       # Pre-trained ML fraud classification model
│   ├── evaluate_offline.py   # Script for batch evaluation metrics
│   ├── simulate_batch.py     # Data ingestion & batch simulation script
│   ├── train_model.py        # Model training lifecycle script
│   ├── generate_data.py      # Synthetic transaction data generator
│   ├── X_train.csv           # Training feature dataset
│   ├── y_train.csv           # Training target labels
│   ├── transactions.csv      # Sample transaction logs
│   └── evaluation_report.json# Offline evaluation metrics output
├── frontend/
│   └── index.html            # Interactive Risk & Fraud Dashboard UI
├── docs/
│   └── PayShield.jpg         # System architecture diagram
├── requirements.txt          # Python package dependencies
└── README.md                 # Project documentation
