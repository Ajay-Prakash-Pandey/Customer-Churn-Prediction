import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException
from src.components.model_training import ModelTrainer
from src.components.prediction_pipeline import PredictionPipeline


class TrainPipeline:
    """
    End-to-End Model Training Pipeline
    """

    def __init__(self):

        project_root = Path(__file__).resolve().parents[2]
        self.engineered_data_path = (
            project_root / "data" / "processed" / "customer_feature_engineered.csv"
        )
        self.raw_data_path = project_root / "data" / "raw" / "customer.csv"

    def run_pipeline(self):

        try:

            logger.info("=" * 60)
            logger.info("MODEL TRAINING PIPELINE STARTED")
            logger.info("=" * 60)

            if self.engineered_data_path.exists():
                df = pd.read_csv(self.engineered_data_path)
            else:
                df = pd.read_csv(self.raw_data_path)

            y = df["Churn"].map({"No": 0, "Yes": 1}).fillna(df["Churn"]).astype(int)

            prediction_pipeline = PredictionPipeline()
            X = prediction_pipeline.prepare_features(
                df.drop(columns=["Churn"])
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            logger.info(f"Train Shape : {X_train.shape}")
            logger.info(f"Test Shape  : {X_test.shape}")

            # Train model
            trainer = ModelTrainer()

            result = trainer.train_from_feature_frames(
                X_train,
                X_test,
                y_train,
                y_test
            )

            logger.info("=" * 60)
            logger.info("MODEL TRAINING PIPELINE COMPLETED")
            logger.info("=" * 60)

            return result

        except Exception as e:

            logger.exception("Training Pipeline Failed")

            raise CustomException(e, sys)


if __name__ == "__main__":

    pipeline = TrainPipeline()

    result = pipeline.run_pipeline()

    print("\n========== TRAINING RESULT ==========")
    print(f"Best Model    : {result['best_model_name']}")
    print(f"Test Accuracy : {result['test_accuracy']:.4f}")
    print(f"Model Saved   : {result['model_path']}")
    print("=====================================")
