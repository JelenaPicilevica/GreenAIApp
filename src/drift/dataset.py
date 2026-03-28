"""
Dataset loading with CORRECT drift definition:
- drift = 1 → sentences are NOT paraphrases (distribution shift)
- drift = 0 → paraphrases (normal data)
"""

from datasets import load_dataset


class DatasetLoader:

    def __init__(self):
        self.name = "paws"
        self.config = "labeled_final"

    def _add_drift_column(self, df, split_name):
        """
        Correct drift definition:
        - If NOT paraphrase (label=0) → drift=1
        - If paraphrase (label=1) → drift=0
        """
        df = df.copy()

        # Core logic
        df["drift"] = (df["label"] == 0).astype(int)

        print(f"\n[{split_name}] Drift stats:")
        print("Total samples:", len(df))
        print("Drift=1 (non-paraphrase):", df["drift"].sum())
        print("Drift=0 (paraphrase):", (df["drift"] == 0).sum())

        return df

    def _preview(self, df, name):
        """
        Show real examples with drift signal.
        """
        print(f"\n===== {name} SAMPLE =====")

        for i in range(2):
            row = df.iloc[i]

            print(f"\nExample {i}:")
            print("Sentence1:", row["sentence1"])
            print("Sentence2:", row["sentence2"])
            print("Label:", row["label"])
            print("Drift:", row["drift"])

    def load(self):

        print("\nLoading dataset...")

        dataset = load_dataset(self.name, self.config)

        train_df = dataset["train"].to_pandas()
        val_df = dataset["validation"].to_pandas()
        test_df = dataset["test"].to_pandas()

        print("Sizes:", len(train_df), len(val_df), len(test_df))

        # Apply correct drift logic
        train_df = self._add_drift_column(train_df, "TRAIN")
        val_df = self._add_drift_column(val_df, "VAL")
        test_df = self._add_drift_column(test_df, "TEST")

        # Preview
        self._preview(train_df, "TRAIN")
        self._preview(val_df, "VAL")
        self._preview(test_df, "TEST")

        return train_df, val_df, test_df