"""
Smart Lender - Flask Web Application
Serves a form for loan applicant details and returns a real-time
creditworthiness / loan approval prediction using the trained model.
"""

import os
import json
import joblib
import numpy as np
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)

# Load trained artifacts once at startup
model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))

with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
    metrics = json.load(f)

CATEGORICAL_COLS = ["Gender", "Married", "Dependents", "Education",
                     "Self_Employed", "Property_Area"]


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", metrics=metrics)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form

        raw = {
            "Gender": form.get("gender"),
            "Married": form.get("married"),
            "Dependents": form.get("dependents"),
            "Education": form.get("education"),
            "Self_Employed": form.get("self_employed"),
            "ApplicantIncome": float(form.get("applicant_income")),
            "CoapplicantIncome": float(form.get("coapplicant_income")),
            "LoanAmount": float(form.get("loan_amount")),
            "Loan_Amount_Term": float(form.get("loan_term")),
            "Credit_History": float(form.get("credit_history")),
            "Property_Area": form.get("property_area"),
        }

        # Encode categorical fields with the fitted LabelEncoders
        row = []
        for col in feature_cols:
            if col in CATEGORICAL_COLS:
                le = encoders[col]
                val = str(raw[col])
                if val not in le.classes_:
                    val = le.classes_[0]  # fallback to a known class
                row.append(le.transform([val])[0])
            else:
                row.append(raw[col])

        X = np.array(row).reshape(1, -1)
        pred = model.predict(X)[0]

        target_le = encoders["Loan_Status"]
        label = target_le.inverse_transform([pred])[0]
        approved = (label == "Y")

        # Confidence score, if the model supports probability estimates
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            confidence = round(float(np.max(proba)) * 100, 1)

        return render_template(
            "result.html",
            approved=approved,
            confidence=confidence,
            inputs=raw,
            model_name=metrics["best_model"],
        )

    except Exception as e:
        return render_template("result.html", error=str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
