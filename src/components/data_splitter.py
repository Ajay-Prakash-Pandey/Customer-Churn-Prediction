from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import PROCESSED_DATA_DIR


class DataSplitter:

    def split_data(self):

        print("=" * 60)
        print("DATA SPLITTING")
        print("=" * 60)

        df = pd.read_csv(
            PROCESSED_DATA_DIR / "customer_final.csv"
        )

        # Features and Target
        X = df.drop("Churn", axis=1)
        y = df["Churn"]

        # 80% Train | 20% Test
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        # 80% of train | 20% of train = Validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.20,
            random_state=42,
            stratify=y_train
        )

        train_df = X_train.copy()
        train_df["Churn"] = y_train

        val_df = X_val.copy()
        val_df["Churn"] = y_val

        test_df = X_test.copy()
        test_df["Churn"] = y_test

        train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
        val_df.to_csv(PROCESSED_DATA_DIR / "validation.csv", index=False)
        test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

        print(f"Train Shape      : {train_df.shape}")
        print(f"Validation Shape : {val_df.shape}")
        print(f"Test Shape       : {test_df.shape}")

        print("\nData Split Completed.")