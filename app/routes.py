import json

import pandas as pd
from flask import Blueprint, flash, redirect, render_template, request, url_for

from config import REPORT_DIR
from src.components.prediction_pipeline import PredictionPipeline

main = Blueprint(
    "main",
    __name__
)


FORM_OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
}

NUMERIC_FIELDS = ["tenure", "MonthlyCharges", "TotalCharges"]


@main.route("/")
def home():
    return render_template("index.html", metrics=_load_metrics())


@main.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template(
            "predict.html",
            options=FORM_OPTIONS,
            values=_default_form_values()
        )

    try:
        customer_data = _form_to_customer_data(request.form)
        pipeline = PredictionPipeline()
        prediction, probability = pipeline.predict(customer_data)
        churn_probability = _extract_churn_probability(prediction, probability)

        return render_template(
            "result.html",
            prediction=int(prediction[0]),
            probability=churn_probability,
            customer=customer_data
        )
    except Exception as exc:
        flash(str(exc), "danger")
        return render_template(
            "predict.html",
            options=FORM_OPTIONS,
            values=request.form
        ), 400


@main.route("/batch", methods=["GET", "POST"])
def batch_prediction():
    if request.method == "GET":
        return render_template("batch_prediction.html", rows=None)

    upload = request.files.get("file")

    if not upload or upload.filename == "":
        flash("Please upload a CSV file.", "warning")
        return redirect(url_for("main.batch_prediction"))

    try:
        df = pd.read_csv(upload)
        pipeline = PredictionPipeline()
        predictions, probabilities = pipeline.predict(df)

        result_df = df.copy()
        result_df["ChurnPrediction"] = [
            "Yes" if value == 1 else "No" for value in predictions
        ]

        if probabilities is not None:
            result_df["ChurnProbability"] = probabilities[:, 1].round(4)

        return render_template(
            "batch_prediction.html",
            rows=result_df.head(50).to_dict(orient="records"),
            columns=result_df.columns.tolist(),
            row_count=len(result_df)
        )
    except Exception as exc:
        flash(str(exc), "danger")
        return redirect(url_for("main.batch_prediction"))


@main.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", metrics=_load_metrics())


@main.route("/train", methods=["POST"])
def train():
    try:
        metrics = PredictionPipeline().train_application_model()
        flash(
            f"Model trained successfully with accuracy {metrics['accuracy']}.",
            "success"
        )
    except Exception as exc:
        flash(str(exc), "danger")

    return redirect(url_for("main.dashboard"))


@main.route("/about")
def about():
    return render_template("about.html")


def _form_to_customer_data(form):
    data = {}

    for field in FORM_OPTIONS:
        value = form.get(field)

        if field == "SeniorCitizen":
            data[field] = int(value)
        else:
            data[field] = value

    for field in NUMERIC_FIELDS:
        data[field] = float(form.get(field, 0))

    return data


def _default_form_values():
    values = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.35,
        "TotalCharges": 904.2
    }

    return values


def _extract_churn_probability(prediction, probability):
    if probability is None:
        return None

    churn_index = 1

    return round(float(probability[0][churn_index]), 4)


def _load_metrics():
    metrics_path = REPORT_DIR / "app_metrics.json"

    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    return {
        "model_name": "Not trained for app yet",
        "accuracy": None,
        "training_rows": None,
        "test_rows": None,
        "feature_count": None
    }
