import os
import json

from pathlib import Path
import numpy as np
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import accuracy_score

from src.logger import logger
from src.exception import CustomException
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    project_root: Path = Path(__file__).resolve().parents[2]
    trained_model_file_path: str = str(project_root / "artifacts" / "best_model.pkl")
    feature_columns_file_path: str = str(project_root / "artifacts" / "feature_columns.json")
    metrics_file_path: str = str(project_root / "reports" / "app_metrics.json")


class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(
        self,
        train_array,
        test_array
    ):

        try:

            logger.info("Starting Model Training...")

            X_train = np.asarray(train_array[:, :-1], dtype=np.float64)
            y_train = np.asarray(train_array[:, -1], dtype=np.int64)

            X_test = np.asarray(test_array[:, :-1], dtype=np.float64)
            y_test = np.asarray(test_array[:, -1], dtype=np.int64)

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            }

            params = {

                "Logistic Regression": {
                    "C": [0.1, 1, 10]
                },

                "Random Forest": {
                    "n_estimators": [100],
                    "max_depth": [None, 10]
                },

                "Gradient Boosting": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [100]
                }

            }

            logger.info("Evaluating models...")

            model_report = evaluate_models(

                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params

            )
            best_model_score = max(model_report.values())

            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model = models[best_model_name]

            logger.info(f"Best Model: {best_model_name}")
            logger.info(f"Best Accuracy: {best_model_score:.4f}")

            # Train the selected model on the complete training data
            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logger.info(
                f"Model saved to {self.model_trainer_config.trained_model_file_path}"
            )

            test_accuracy = best_model.score(
                X_test,
                y_test
            )

            logger.info(
                f"Final Test Accuracy: {test_accuracy:.4f}"
            )

            return {
                "best_model_name": best_model_name,
                "test_accuracy": test_accuracy,
                "model_path": self.model_trainer_config.trained_model_file_path
            }

        except Exception as e:

            logger.exception("Model training failed.")

            raise CustomException(e, sys)

    def train_from_feature_frames(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):
        """
        Train the app model from named feature dataframes and save the
        metadata required by the Flask prediction pipeline.
        """

        try:
            logger.info("Starting named-feature model training...")

            X_train_values = np.asarray(X_train, dtype=np.float64)
            X_test_values = np.asarray(X_test, dtype=np.float64)
            y_train_values = np.asarray(y_train, dtype=np.int64)
            y_test_values = np.asarray(y_test, dtype=np.int64)

            model = GradientBoostingClassifier(random_state=42)
            model.fit(X_train_values, y_train_values)

            y_pred = model.predict(X_test_values)
            test_accuracy = accuracy_score(y_test_values, y_pred)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            feature_columns_path = Path(
                self.model_trainer_config.feature_columns_file_path
            )
            feature_columns_path.parent.mkdir(parents=True, exist_ok=True)
            feature_columns_path.write_text(
                json.dumps(list(X_train.columns), indent=2),
                encoding="utf-8"
            )

            metrics = {
                "model_name": "GradientBoostingClassifier",
                "accuracy": round(float(test_accuracy), 4),
                "training_rows": int(X_train.shape[0]),
                "test_rows": int(X_test.shape[0]),
                "feature_count": int(X_train.shape[1])
            }

            metrics_path = Path(self.model_trainer_config.metrics_file_path)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                json.dumps(metrics, indent=2),
                encoding="utf-8"
            )

            logger.info(f"App model accuracy: {test_accuracy:.4f}")

            return {
                "best_model_name": metrics["model_name"],
                "test_accuracy": test_accuracy,
                "model_path": self.model_trainer_config.trained_model_file_path,
                "feature_columns_path": str(feature_columns_path),
                "metrics_path": str(metrics_path)
            }

        except Exception as e:
            logger.exception("Named-feature model training failed.")
            raise CustomException(e, sys)
