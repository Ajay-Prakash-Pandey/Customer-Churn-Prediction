from config import RAW_DATA_PATH

from src.components.data_loader import DataLoader
from src.components.data_cleaning import DataCleaner
from src.components.feature_engineering import FeatureEngineer
from src.components.preprocessing import DataPreprocessor


loader = DataLoader(RAW_DATA_PATH)

df = loader.load_data()

cleaner = DataCleaner()

df = cleaner.clean(df)

engineer = FeatureEngineer()

df = engineer.create_features(df)

preprocessor = DataPreprocessor()

final_df = preprocessor.transform(df)

print("\nFinal Shape")

print(final_df.shape)

print("\nColumns")

print(final_df.columns.tolist()[:15])
