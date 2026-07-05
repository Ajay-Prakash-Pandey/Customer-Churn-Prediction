from config import RAW_DATA_PATH
# from src.components.data_loader import DataLoader
from src.components.data_loader import DataLoader
loader = DataLoader(RAW_DATA_PATH)

df = loader.load_data()

print(df.head())

print("\nShape:", df.shape)

print("\nColumns:")

print(df.columns.tolist())
