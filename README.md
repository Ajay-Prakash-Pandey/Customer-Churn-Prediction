# Customer Churn Prediction

A Flask-based machine learning web application that predicts whether a telecom customer is likely to churn. The project includes data processing, feature engineering, model training/evaluation, single-customer prediction, batch CSV prediction, a dashboard, tests, and deployment configuration.

## Project Overview

Customer churn is a major business problem for telecom companies because retaining an existing customer is usually cheaper than acquiring a new one. This project uses the IBM Telco Customer Churn dataset to train a machine learning model that predicts churn risk from customer account, service, billing, and contract details.

The application can be used to:

- Predict churn for one customer through a web form.
- Upload a CSV file and generate batch predictions.
- View model metrics on a dashboard.
- Retrain the application model from available processed/raw data.
- Deploy the Flask app to a cloud hosting platform such as Render.

## Features

- Flask web interface
- Single customer churn prediction
- Batch CSV prediction
- Dashboard with model metrics
- Data cleaning and preprocessing pipeline
- Feature engineering pipeline
- Model training and evaluation modules
- Saved runtime artifacts for production predictions
- Organized test suite inside `testing/`
- Render deployment files included

## Dataset

Dataset used: IBM Telco Customer Churn Dataset

Expected raw dataset path:

```text
data/raw/customer.csv
```

## Tech Stack

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Gunicorn
- HTML/CSS templates

## Project Structure

```text
Customer Churn Prediction/
|-- app/
|   |-- static/
|   |-- templates/
|   |-- __init__.py
|   |-- forms.py
|   |-- routes.py
|   `-- utils.py
|-- artifacts/
|   |-- best_model.pkl
|   |-- evaluation.json
|   `-- feature_columns.json
|-- data/
|   |-- raw/
|   `-- processed/
|-- docs/
|-- logs/
|-- models/
|-- reports/
|   `-- app_metrics.json
|-- src/
|   |-- components/
|   |-- pipeline/
|   |-- custom_data.py
|   |-- exception.py
|   |-- logger.py
|   `-- utils.py
|-- testing/
|   `-- test_*.py
|-- app.py
|-- config.py
|-- predict.py
|-- wsgi.py
|-- requirements.txt
|-- Procfile
|-- render.yaml
|-- runtime.txt
|-- DEPLOYMENT.md
`-- README.md
```

## Important Runtime Files

The deployed app needs these files:

```text
artifacts/best_model.pkl
artifacts/feature_columns.json
reports/app_metrics.json
```

These files are allowed in `.gitignore` so they can be committed with the project.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

Start the Flask development server:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Production Entry Point

The production WSGI entry point is:

```text
wsgi:app
```

Production start command:

```bash
gunicorn wsgi:app
```

On Windows, use `python app.py` for local development because Gunicorn is mainly used on Linux hosting environments.

## App Routes

| Route | Description |
| --- | --- |
| `/` | Home page with model summary |
| `/predict` | Single customer prediction form |
| `/batch` | Batch CSV prediction upload |
| `/dashboard` | Application metrics dashboard |
| `/train` | Retrain model from available data |
| `/about` | Project/about page |

## Testing

All test files are stored in:

```text
testing/
```

Run tests:

```bash
pytest testing
```

If pytest is not installed:

```bash
pip install pytest
```

## Deployment

This project includes deployment files for Render:

- `Procfile`
- `render.yaml`
- `runtime.txt`
- `wsgi.py`
- `DEPLOYMENT.md`

Render settings:

```text
Build Command: pip install -r requirements.txt
Start Command: gunicorn wsgi:app
Python Version: 3.12.8
```

Set this environment variable in Render:

```text
SECRET_KEY=your-long-random-secret
```

Full deployment steps are available in `DEPLOYMENT.md`.

## Model Workflow

1. Load raw customer data.
2. Clean missing and inconsistent values.
3. Engineer customer behavior and billing features.
4. Encode categorical variables.
5. Train and evaluate classification models.
6. Save the best model and feature columns.
7. Serve predictions through the Flask app.

## Git Notes

The `.gitignore` file excludes virtual environments, caches, logs, editor files, generated processed data, and large model folders. It keeps the required deployment artifacts trackable.

If Git is not initialized correctly, run:

```bash
git init
git add .
git commit -m "Initial customer churn prediction app"
```
