"""
Smart Lender - Model Training Script
Trains Decision Tree, Random Forest, KNN, and XGBoost classifiers on the
loan applicant dataset, evaluates each, and saves the best-performing model
(by testing accuracy) along with the fitted preprocessing objects for use
in the Flask web application.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

CATEGORICAL_COLS = ["Gender", "Married", "Dependents", "Education",
                     "Self_Employed", "Property_Area"]
NUMERIC_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                 "Loan_Amount_Term", "Credit_History"]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS
TARGET_COL = "Loan_Status"


def load_and_clean_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Loan_ID"])

    # Normalize "3+" dependents to 3 for numeric handling later, keep as category for encoding
    df["Dependents"] = df["Dependents"].astype(str).str.replace("+", "", regex=False)

    # Fill missing values
    for col in ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])

    return df


def encode_features(df):
    encoders = {}
    df_encoded = df.copy()
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le

    target_le = LabelEncoder()
    df_encoded[TARGET_COL] = target_le.fit_transform(df_encoded[TARGET_COL].astype(str))
    encoders[TARGET_COL] = target_le

    return df_encoded, encoders


def train_and_evaluate():
    df = load_and_clean_data()
    df_encoded, encoders = encode_features(df)

    X = df_encoded[FEATURE_COLS]
    y = df_encoded[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=42
        ),
    }

    results = {}
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = {"train_accuracy": round(train_acc * 100, 2),
                          "test_accuracy": round(test_acc * 100, 2)}
        fitted_models[name] = model
        print(f"{name:15s} | Train Acc: {train_acc*100:6.2f}% | Test Acc: {test_acc*100:6.2f}%")

    best_name = max(results, key=lambda k: results[k]["test_accuracy"])
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name} "
          f"(Train {results[best_name]['train_accuracy']}% / Test {results[best_name]['test_accuracy']}%)")

    # Persist model, encoders, feature order, and metrics
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
    joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, "feature_cols.pkl"))

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)

    return best_name, results


if __name__ == "__main__":
    train_and_evaluate()
