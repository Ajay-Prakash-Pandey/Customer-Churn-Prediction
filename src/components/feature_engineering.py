import os
import pandas as pd

from config import PROCESSED_DATA_DIR


class FeatureEngineer:
    """
    Create new features for Customer Churn Prediction
    """

    def create_features(self, df):

        print("=" * 50)
        print("Feature Engineering Started")
        print("=" * 50)

        # Total Services
        service_columns = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "PhoneService"
        ]

        df["TotalServices"] = (
            df[service_columns] == "Yes"
        ).sum(axis=1)

        # Average Monthly Spend
        df["AvgMonthlySpend"] = (
            df["TotalCharges"] / (df["tenure"] + 1)
        )

        # High Value Customer
        df["HighValueCustomer"] = (
            df["MonthlyCharges"] > 70
        ).astype(int)

        # Long Term Customer
        df["LongTermCustomer"] = (
            df["tenure"] >= 24
        ).astype(int)

        # Streaming User
        df["StreamingUser"] = (
            (
                (df["StreamingTV"] == "Yes")
                |
                (df["StreamingMovies"] == "Yes")
            )
        ).astype(int)

        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        output_path = PROCESSED_DATA_DIR / "customer_feature_engineered.csv"

        df.to_csv(output_path, index=False)

        print(f"\nSaved: {output_path}")

        print("\nFeature Engineering Completed")

        return df