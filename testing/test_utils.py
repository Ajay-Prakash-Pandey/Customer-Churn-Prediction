import pandas as pd

from config import ARTIFACTS_DIR
from src.utils import save_object
from src.utils import load_object


# Sample DataFrame

df = pd.DataFrame({

    "Name": ["Ajay", "Rahul"],

    "Age": [22, 23]

})


# Save object

save_object(
    ARTIFACTS_DIR / "test_dataframe.pkl",
    df
)

print("Object Saved Successfully")


# Load object

loaded_df = load_object(
    ARTIFACTS_DIR / "test_dataframe.pkl"
)

print("\nLoaded DataFrame")

print(loaded_df)
