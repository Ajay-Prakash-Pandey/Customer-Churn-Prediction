"""
data_loader.py
----------------
Load the raw Telco Customer Churn dataset.
"""

from pathlib import Path
import pandas as pd


class DataLoader:
    """
    Loads the customer churn dataset.
    """

    def __init__(self, data_path):
        self.data_path = Path(data_path)

    def load_data(self):
        """
        Returns
        -------
        pandas.DataFrame
        """

        # if not self.data_path.exists():
        #     raise FileNotFoundError(
        #         f"Dataset not found:\n{self.data_path}"
        #     )

        df = pd.read_csv(self.data_path)

        return df