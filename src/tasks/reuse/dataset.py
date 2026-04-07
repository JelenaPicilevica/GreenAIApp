# reuse/neuralNetwork/dataset.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split


class ReuseDatasetLoader:

    def __init__(self, path=None):

        if path is None:
            BASE_DIR = os.path.dirname(__file__)
            path = os.path.join(BASE_DIR, "dataset/dataset_25k.csv")

        self.path = os.path.abspath(path)

    def load(self):

        print("\nLoading REUSE dataset...")
        print("Path:", self.path)

        df = pd.read_csv(self.path)

        df = df.rename(columns={
            "prompt1": "sentence1",
            "prompt2": "sentence2"
        })

        # 🔥 КЛЮЧЕВОЕ
        df["drift"] = df["label"]

        print("\n[DATASET STATS]")
        print("Total:", len(df))
        print(df["label"].value_counts(normalize=True))

        # stratified split
        train, temp = train_test_split(
            df,
            test_size=0.2,
            stratify=df["label"],
            random_state=42
        )

        val, test = train_test_split(
            temp,
            test_size=0.5,
            stratify=temp["label"],
            random_state=42
        )

        print("\n[SPLITS]")
        print("Train:", len(train))
        print("Val:", len(val))
        print("Test:", len(test))

        return train, val, test