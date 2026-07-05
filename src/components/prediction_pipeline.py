import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException
from src.utils import load_object, save_object


class PredictionPipeline:

    def __init__(self):

        self.project_root = Path(__file__).resolve().parents[2]
        self.model_path = self.project_root / "artifacts" / "best_model.pkl"
        self.feature_columns_path = (
            self.project_root / "artifacts" / "feature_columns.json"
        )
        self.metrics_path = self.project_root / "reports" / "app_metrics.json"
        self.processed_data_path = (
            self.project_root / "data" / "processed" / "customer_final.csv"
        )

        self.feature_columns = self._load_feature_columns()
        self.model = self._load_or_train_model()

    def predict(self, features):

        try:

            logger.info("Prediction Started.")

            prepared_features = self.prepare_features(features)

            prediction = self.model.predict(prepared_features)

            probability = None

            if hasattr(self.model, "predict_proba"):

                probability = self.model.predict_proba(
                    prepared_features
                )

            logger.info("Prediction Completed.")

            return prediction, probability

        except Exception as e:

            logger.exception("Prediction Failed.")

            raise CustomException(
                e,
                sys
            )

    def prepare_features(self, features):

        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features.copy()

        df = self._clean_input(df)
        df = self._create_features(df)
        df = pd.get_dummies(df, drop_first=True)

        for column in self.feature_columns:
            if column not in df.columns:
                df[column] = 0

        df = df[self.feature_columns]

        return df.astype(float)

    def train_application_model(self):

        if not self.processed_data_path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {self.processed_data_path}"
            )

        training_df = self._load_training_dataframe()
        y = training_df["Churn"].map({"No": 0, "Yes": 1}).fillna(
            training_df["Churn"]
        ).astype(int)
        X = self.prepare_features(training_df.drop(columns=["Churn"]))

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        save_object(str(self.model_path), model)

        self.feature_columns = X.columns.tolist()
        self.feature_columns_path.parent.mkdir(parents=True, exist_ok=True)
        self.feature_columns_path.write_text(
            json.dumps(self.feature_columns, indent=2),
            encoding="utf-8"
        )

        metrics = {
            "model_name": "GradientBoostingClassifier",
            "accuracy": round(float(accuracy), 4),
            "training_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "feature_count": int(X.shape[1])
        }
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8"
        )

        self.model = model

        return metrics

    def _load_training_dataframe(self):

        engineered_data_path = (
            self.project_root / "data" / "processed" / "customer_feature_engineered.csv"
        )
        raw_data_path = self.project_root / "data" / "raw" / "customer.csv"

        if engineered_data_path.exists():
            return pd.read_csv(engineered_data_path)

        if raw_data_path.exists():
            return pd.read_csv(raw_data_path)

        raise FileNotFoundError(
            "Training data was not found in data/processed or data/raw."
        )

    def _load_feature_columns(self):

        if self.feature_columns_path.exists():
            return json.loads(
                self.feature_columns_path.read_text(encoding="utf-8")
            )

        if self.processed_data_path.exists():
            df = pd.read_csv(self.processed_data_path, nrows=1)
            return df.drop(columns=["Churn"]).columns.tolist()

        raise FileNotFoundError(
            "Feature columns could not be loaded. Run training first."
        )

    def _load_or_train_model(self):

        should_train = not self.model_path.exists()

        if not should_train:
            model = load_object(str(self.model_path))
            expected_features = getattr(model, "n_features_in_", None)
            should_train = expected_features != len(self.feature_columns)

            if not should_train:
                return model

            logger.warning(
                "Model feature count mismatch. Retraining application model."
            )

        self.train_application_model()

        return self.model

    def _clean_input(self, df):

        df = df.copy()

        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])

        if "Churn" in df.columns:
            df = df.drop(columns=["Churn"])

        numeric_columns = [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

        return df

    def _create_features(self, df):

        df = df.copy()

        service_columns = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "PhoneService"
        ]

        df["TotalServices"] = (df[service_columns] == "Yes").sum(axis=1)
        df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
        df["HighValueCustomer"] = (df["MonthlyCharges"] > 70).astype(int)
        df["LongTermCustomer"] = (df["tenure"] >= 24).astype(int)
        df["StreamingUser"] = (
            (df["StreamingTV"] == "Yes")
            | (df["StreamingMovies"] == "Yes")
        ).astype(int)

        return df
