import os
import pandas as pd

from config import PROCESSED_DATA_DIR


class DataPreprocessor:

    def transform(self, df):

        print("=" * 50)
        print("PREPROCESSING STARTED")
        print("=" * 50)

        # -----------------------------------
        # Drop customerID
        # -----------------------------------

        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])

        # -----------------------------------
        # Encode Target
        # -----------------------------------

        df["Churn"] = df["Churn"].map({
            "No": 0,
            "Yes": 1
        })

        # -----------------------------------
        # One-Hot Encoding
        # -----------------------------------

        df = pd.get_dummies(
            df,
            drop_first=True
        )

        if "Churn" in df.columns:
            feature_columns = [
                column for column in df.columns if column != "Churn"
            ]
            df = df[feature_columns + ["Churn"]]

        # -----------------------------------
        # Save Final Dataset
        # -----------------------------------

        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        output_path = PROCESSED_DATA_DIR / "customer_final.csv"

        df.to_csv(
            output_path,
            index=False
        )

        print(f"\nSaved : {output_path}")

        print(f"\nDataset Shape : {df.shape}")

        print("\nPREPROCESSING COMPLETED")

        return df
