import numpy as np


class DriftRouter:
    """
    Combines neural network predictions with transformer fallback
    by routing uncertain samples to the transformer model.
    """

    def __init__(self, transformer, low=0.3, high=0.7):
        self.transformer = transformer
        self.low = low
        self.high = high

    def route(self, probs, df):
        print("\n===== HYBRID =====")

        mid_zone = (probs > self.low) & (probs < self.high)

        print("Transformer usage:", mid_zone.mean())

        final_probs = probs.copy()

        for i in np.where(mid_zone)[0]:
            s1 = df["sentence1"].iloc[i]
            s2 = df["sentence2"].iloc[i]
            _, p = self.transformer.is_drift(s1, s2)
            final_probs[i] = p

        return final_probs