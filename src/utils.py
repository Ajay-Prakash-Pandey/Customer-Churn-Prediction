import os
import sys
import pickle
from pathlib import Path

import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from src.logger import logger
from src.exception import CustomException


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(file_path):
    path = Path(file_path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def save_object(file_path: str, obj):
    """
    Save Python object.
    """

    try:

        file_path = resolve_project_path(file_path)

        directory = file_path.parent

        os.makedirs(directory, exist_ok=True)

        with open(file_path, "wb") as file:

            pickle.dump(obj, file)

        logger.info(f"Object saved at {file_path}")

    except Exception as e:

        raise CustomException(e, sys)


def load_object(file_path: str):
    """
    Load Python object.
    """

    try:

        file_path = resolve_project_path(file_path)

        with open(file_path, "rb") as file:

            obj = pickle.load(file)

        logger.info(f"Object loaded from {file_path}")

        return obj

    except Exception as e:

        raise CustomException(e, sys)


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    params
):
    """
    Train and evaluate multiple models.
    """

    try:

        # Convert dataframe/object arrays to numeric arrays
        X_train = np.asarray(X_train).astype(float)
        X_test = np.asarray(X_test).astype(float)

        y_train = np.asarray(y_train).astype(int)
        y_test = np.asarray(y_test).astype(int)

        report = {}

        for model_name, model in models.items():

            param = params.get(model_name, {})

            if param:

                grid = GridSearchCV(
                    estimator=model,
                    param_grid=param,
                    cv=5,
                    scoring="accuracy",
                    n_jobs=-1,
                    error_score="raise"
                )

                grid.fit(X_train, y_train)

                best_model = grid.best_estimator_

            else:

                best_model = model

                best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            report[model_name] = accuracy

            logger.info(
                f"{model_name} Accuracy : {accuracy:.4f}"
            )

            # Update the original dictionary with the fitted model
            models[model_name] = best_model

        return report

    except Exception as e:

        raise CustomException(e, sys)
