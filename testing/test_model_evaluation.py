import pandas as pd

from src.utils import load_object
from src.components.model_evaluation import ModelEvaluator


def test_model_evaluation():

    print("\n========== MODEL EVALUATION TEST ==========\n")

    # Load trained model
    model = load_object("artifacts/best_model.pkl")

    # Load test dataset
    test_df = pd.read_csv("data/processed/test.csv")

    # Split features and target
    X_test = test_df.iloc[:, :-1].astype(float).values
    y_test = test_df.iloc[:, -1].astype(int).values

    # Evaluate model
    evaluator = ModelEvaluator()

    results = evaluator.evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test
    )

    print("Evaluation Results\n")

    for metric, value in results.items():
        print(f"{metric}: {value}")

    print("\n===========================================")


if __name__ == "__main__":
    test_model_evaluation()