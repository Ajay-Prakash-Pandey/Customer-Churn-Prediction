from config import RAW_DATA_PATH

from src.components.data_loader import DataLoader
from src.components.data_cleaning import DataCleaner


loader = DataLoader(RAW_DATA_PATH)

df = loader.load_data()

cleaner = DataCleaner()

clean_df = cleaner.clean(df)

print("\n")

print(clean_df.head())

print("\nShape")

print(clean_df.shape)

print("\nData Types")

print(clean_df.dtypes)
