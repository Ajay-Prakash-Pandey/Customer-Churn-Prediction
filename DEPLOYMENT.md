# Deployment

This Flask app is ready to deploy as a Python web service.

## Render

1. Push this project to GitHub.
2. Create a new Render Web Service from the GitHub repository.
3. Use these settings:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn wsgi:app`
   - Python version: `3.12.8`
4. Add an environment variable:
   - `SECRET_KEY`: any long random string
5. Deploy.

The app needs these runtime files in the repository:

- `artifacts/best_model.pkl`
- `artifacts/feature_columns.json`
- `reports/app_metrics.json`

## Local Production Check

Run the same server command locally:

```bash
gunicorn wsgi:app
```

On Windows, use the normal Flask runner for local development:

```bash
python app.py
```
