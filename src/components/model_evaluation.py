import os
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.logger import logger
from src.exception import CustomException
from src.utils import load_object
from config import ARTIFACTS_DIR


@dataclass
class ModelEvaluationConfig:
    evaluation_file_path = ARTIFACTS_DIR / "evaluation.json"


class ModelEvaluator:

    def __init__(self):

        self.config = ModelEvaluationConfig()

    def evaluate_model(
        self,
        model,
        X_test,
        y_test
    ):

        try:

            logger.info("Starting Model Evaluation...")

            y_pred = model.predict(X_test)

            results = {

                "accuracy": float(
                    accuracy_score(y_test, y_pred)
                ),

                "precision": float(
                    precision_score(
                        y_test,
                        y_pred,
                        zero_division=0
                    )
                ),

                "recall": float(
                    recall_score(
                        y_test,
                        y_pred,
                        zero_division=0
                    )
                ),

                "f1_score": float(
                    f1_score(
                        y_test,
                        y_pred,
                        zero_division=0
                    )
                )

            }

            try:

                y_prob = model.predict_proba(X_test)[:, 1]

                results["roc_auc"] = float(

                    roc_auc_score(
                        y_test,
                        y_prob
                    )

                )

            except Exception:

                results["roc_auc"] = None

            results["confusion_matrix"] = confusion_matrix(
                y_test,
                y_pred
            ).tolist()

            results["classification_report"] = classification_report(
                y_test,
                y_pred,
                output_dict=True,
                zero_division=0
            )

            Path(self.config.evaluation_file_path).parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with open(
                self.config.evaluation_file_path,
                "w"
            ) as file:

                json.dump(
                    results,
                    file,
                    indent=4
                )

            logger.info("Model Evaluation Completed Successfully.")

            return results

        except Exception as e:

            logger.exception("Error during Model Evaluation.")

            raise CustomException(
                e,
                sys
            )

    def evaluate(self):
        """
        Evaluate the current application model using the same feature
        preparation contract as the Flask prediction pipeline.
        """

        try:
            from src.components.prediction_pipeline import PredictionPipeline

            pipeline = PredictionPipeline()
            df = pipeline._load_training_dataframe()
            y = df["Churn"].map({"No": 0, "Yes": 1}).fillna(
                df["Churn"]
            ).astype(int)
            X = pipeline.prepare_features(df.drop(columns=["Churn"]))

            _, X_test, _, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            model = load_object(pipeline.model_path)

            return self.evaluate_model(
                model,
                X_test,
                y_test
            )

        except Exception as e:
            logger.exception("Default model evaluation failed.")
            raise CustomException(e, sys)
