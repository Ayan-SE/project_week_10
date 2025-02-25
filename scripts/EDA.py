import pandas as pd
import numpy as np
from sklearn.preprocessing import  StandardScaler


def handle_missing_values(df):
    """
    Identifies and handles missing values by filling or dropping them.
    """
    missing_count = df.isnull().sum()
    print("🔍 Missing Values Before Handling:\n", missing_count)

    # Fill missing values with forward-fill (ffill) and backward-fill (bfill)
    df.dropna(inplace=True)

    print("✅ Missing values handled successfully.\n")
    return df

# ------------------ 2. Descriptive Statistics ------------------
def summarize_data(df):
    """
    Generates basic descriptive statistics for the dataset.
    """
    print("\n📊 Summary Statistics:\n", df.describe())
    print("\n🛠 Data Info:\n")
    print(df.info())

def standard_scaling(data):
    """
    Performs Standard scaling (Z-score normalization) on the input data.
    """
    scaler = StandardScaler()  # Create a StandardScaler object
    scaled_data = scaler.fit_transform(data) # Fit and transform the data
    return scaled_data