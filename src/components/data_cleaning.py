import pandas as pd


class DataCleaner:

    """
    Clean Customer Churn Dataset
    """

    def clean(self, df):

        print("=" * 50)
        print("Cleaning Started")
        print("=" * 50)

        # Remove duplicate rows

        before = df.shape[0]

        df = df.drop_duplicates()

        after = df.shape[0]

        print(f"Duplicates Removed : {before-after}")

        # Replace blank spaces

        df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)

        # Convert TotalCharges

        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

        # Fill Missing Values

        df["TotalCharges"] = df["TotalCharges"].fillna(
            df["MonthlyCharges"]
        )

        print("\nMissing Values")

        print(df.isnull().sum())

        print("\nCleaning Completed")

        return df