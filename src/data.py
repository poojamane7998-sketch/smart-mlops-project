from pathlib import Path

import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "breast_cancer.csv"


def load_dataset():

    # Create the data folder if it does not exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load the built-in breast cancer dataset
    dataset = load_breast_cancer(as_frame=True)

    # Convert the dataset into a DataFrame
    frame = dataset.frame.copy()

    # Save the dataset as a CSV file
    frame.to_csv(CSV_PATH, index=False)

    # Return dataset and feature names
    return frame, list(dataset.feature_names)


def split_dataset(test_size=0.2, random_state=42):

    # Load dataset
    frame, feature_names = load_dataset()

    # Separate input features
    X = frame[feature_names]

    # Separate target column
    y = frame["target"]

    # Split dataset into training and testing data
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )