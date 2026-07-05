from config import RAW_DATA_PATH

from src.components.data_loader import DataLoader
from src.components.feature_engineering import FeatureEngineer
from src.components.data_cleaning import DataCleaner

loader = DataLoader(RAW_DATA_PATH)

df = loader.load_data()

cleaner = DataCleaner()

df = cleaner.clean(df)

engineer = FeatureEngineer()

df = engineer.create_features(df)

print("\nShape")

print(df.shape)

print("\nNew Columns")

new_columns = [
    "TotalServices",
    "AvgMonthlySpend",
    "HighValueCustomer",
    "LongTermCustomer",
    "StreamingUser"
]

print(df[new_columns].head())
