"""
Inference Benchmark for GreenAI Framework

This benchmark evaluates three deployment scenarios:

1. 100% Neural Network inference
2. 100% Transformer inference
3. Hybrid deployment:
   - 66.5% Neural Network
   - 33.5% Transformer

The benchmark uses the PAWS validation dataset and measures:
- Average inference time
- Transformer usage
- Relative energy cost

Hybrid metrics are calculated analytically using measured
NN and transformer inference times.
"""

import time
from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset

from src.model.transformer_fallback import TransformerFallback


class ArchitectureBenchmark:

    def __init__(
            self,
            nn_model_path,
            transformer_model_path,
            pipeline_path,
            sample_size=1000
    ):

        self.device = "cpu"

        self.sample_size = sample_size

        # =====================================================
        # LOAD DATASET
        # =====================================================

        dataset = load_dataset(
            "paws",
            "labeled_final"
        )

        validation_df = dataset["validation"].to_pandas()

        # Same drift logic as training
        validation_df["drift"] = (
            validation_df["label"] == 0
        ).astype(int)

        self.samples = validation_df.sample(
            n=sample_size,
            random_state=42
        ).reset_index(drop=True)

        print("\n[Benchmark Dataset]")
        print("Total samples:", len(self.samples))
        print("Drift=1:", self.samples["drift"].sum())
        print("Drift=0:", (self.samples["drift"] == 0).sum())

        # =====================================================
        # LOAD FEATURE PIPELINE
        # =====================================================

        self.feature_pipeline = joblib.load(
            str(pipeline_path)
        )

        # =====================================================
        # LOAD NN MODEL
        # =====================================================

        self.nn_model = torch.load(
            str(nn_model_path),
            map_location=self.device,
            weights_only=False
        )

        self.nn_model.eval()

        # =====================================================
        # LOAD TRANSFORMER
        # =====================================================

        self.transformer = TransformerFallback(
            model_path=str(transformer_model_path)
        )

    # =====================================================
    # NN BENCHMARK
    # =====================================================

    def benchmark_neural_network(self):

        start = time.perf_counter()

        for _, row in self.samples.iterrows():

            df = pd.DataFrame([{
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
                "drift": row["drift"]
            }])

            X, _ = self.feature_pipeline.transform(df)

            if hasattr(X, "toarray"):
                X = X.toarray()

            x = torch.tensor(
                X,
                dtype=torch.float32
            )

            with torch.no_grad():

                _ = F.softmax(
                    self.nn_model(x),
                    dim=1
                )

        end = time.perf_counter()

        avg_time_ms = (
            (end - start) / self.sample_size
        ) * 1000

        return round(avg_time_ms, 2)

    # =====================================================
    # TRANSFORMER BENCHMARK
    # =====================================================

    def benchmark_transformer(self):

        start = time.perf_counter()

        for _, row in self.samples.iterrows():

            _ = self.transformer.predict_drift(
                row["sentence1"],
                row["sentence2"]
            )

        end = time.perf_counter()

        avg_time_ms = (
            (end - start) / self.sample_size
        ) * 1000

        return round(avg_time_ms, 2)

    # =====================================================
    # FINAL COMPARISON TABLE
    # =====================================================

    def run(self):

        print("\nRunning benchmarks...\n")

        # Measure actual inference times
        nn_time = self.benchmark_neural_network()

        transformer_time = self.benchmark_transformer()

        # =====================================================
        # ANALYTICAL HYBRID MODEL
        # =====================================================

        transformer_ratio = 0.335
        nn_ratio = 0.665

        hybrid_time = (
            nn_ratio * nn_time
            + transformer_ratio * transformer_time
        )

        # =====================================================
        # RELATIVE ENERGY COST
        # =====================================================

        transformer_energy = (
            transformer_time / nn_time
        )

        hybrid_energy = (
            hybrid_time / nn_time
        )

        # =====================================================
        # FINAL TABLE
        # =====================================================

        results = pd.DataFrame([
            {
                "Model": "Neural Network",
                "Avg Inference Time (ms)": round(nn_time, 2),
                "Transformer Usage": "0%",
                "Relative Energy Cost": "1.00×"
            },
            {
                "Model": "Transformer",
                "Avg Inference Time (ms)": round(transformer_time, 2),
                "Transformer Usage": "100%",
                "Relative Energy Cost":
                    f"{transformer_energy:.2f}×"
            },
            {
                "Model": "Hybrid",
                "Avg Inference Time (ms)": round(hybrid_time, 2),
                "Transformer Usage": "33.5%",
                "Relative Energy Cost":
                    f"{hybrid_energy:.2f}×"
            }
        ])

        print("\n=== Computational Efficiency Comparison ===\n")

        print(results.to_string(index=False))


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    ROOT_DIR = Path(__file__).resolve().parents[2]

    benchmark = ArchitectureBenchmark(
        nn_model_path=ROOT_DIR / "models" / "drift_model.pt",
        transformer_model_path=ROOT_DIR / "models" / "transformer_model.pt",
        pipeline_path=ROOT_DIR / "models" / "drift_feature_pipeline.pkl",
        sample_size=1000
    )

    benchmark.run()