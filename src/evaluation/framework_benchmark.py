"""
Extended Inference Benchmark for GreenAI Framework

Computational workflow evaluation for:
1. Semantic cache retrieval
2. Full framework pipeline
3. Full LLM inference

Results are printed directly into logs as a structured table.

Supports:
Section 4.4 - Comparative Computational Cost Analysis
"""

import time
from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn.functional as F

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI

from src.model.transformer_fallback import TransformerFallback


# =====================================================
# BENCHMARK
# =====================================================

class FrameworkComputationalBenchmark:

    def __init__(
            self,
            nn_model_path,
            transformer_model_path,
            pipeline_path,
            sample_size=20
    ):

        self.device = "cpu"

        self.sample_size = sample_size

        print("\nLoading benchmark components...")

        # =====================================================
        # OPENAI CLIENT
        # =====================================================

        self.client = OpenAI()

        # =====================================================
        # LOAD DATASET
        # =====================================================

        dataset = load_dataset(
            "paws",
            "labeled_final"
        )

        validation_df = dataset["validation"].to_pandas()

        validation_df["drift"] = (
            validation_df["label"] == 0
        ).astype(int)

        self.samples = validation_df.sample(
            n=sample_size,
            random_state=42
        ).reset_index(drop=True)

        print("\n[Benchmark Dataset]")
        print("Total samples:", len(self.samples))

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
        # EMBEDDING MODEL
        # =====================================================

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print("\nBenchmark components loaded.")

        # =====================================================
        # SIMPLE CACHE
        # =====================================================

        self.simple_cache = {}

        for _, row in self.samples.iterrows():

            self.simple_cache[
                row["sentence1"]
            ] = row["sentence2"]

    # =====================================================
    # CACHE LOOKUP
    # =====================================================

    def benchmark_cache_lookup(self):

        start = time.perf_counter()

        for _, row in self.samples.iterrows():

            _ = self.simple_cache.get(
                row["sentence1"]
            )

        end = time.perf_counter()

        avg_time_ms = (
            (end - start) / self.sample_size
        ) * 1000

        return round(avg_time_ms, 4)

    # =====================================================
    # EMBEDDING GENERATION
    # =====================================================

    def benchmark_embedding_generation(self):

        start = time.perf_counter()

        for _, row in self.samples.iterrows():

            _ = self.embedding_model.encode(
                row["sentence1"]
            )

        end = time.perf_counter()

        avg_time_ms = (
            (end - start) / self.sample_size
        ) * 1000

        return round(avg_time_ms, 3)

    # =====================================================
    # SEMANTIC SIMILARITY
    # =====================================================

    def benchmark_semantic_similarity(self):

        start = time.perf_counter()

        for _, row in self.samples.iterrows():

            emb1 = self.embedding_model.encode(
                row["sentence1"]
            )

            emb2 = self.embedding_model.encode(
                row["sentence2"]
            )

            _ = cosine_similarity(
                [emb1],
                [emb2]
            )

        end = time.perf_counter()

        avg_time_ms = (
            (end - start) / self.sample_size
        ) * 1000

        return round(avg_time_ms, 3)

    # =====================================================
    # FEATURE COMPUTATION
    # =====================================================

    def benchmark_feature_computation(self):

        start = time.perf_counter()

        for _, row in self.samples.iterrows():

            df = pd.DataFrame([{
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
                "drift": row["drift"]
            }])

            _ = self.feature_pipeline.transform(df)

        end = time.perf_counter()

        avg_time_ms = (
            (end - start) / self.sample_size
        ) * 1000

        return round(avg_time_ms, 3)

    # =====================================================
    # NEURAL DRIFT DETECTION
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

        return round(avg_time_ms, 3)

    # =====================================================
    # TRANSFORMER FALLBACK
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

        return round(avg_time_ms, 3)

    # =====================================================
    # FULL LLM INFERENCE
    # =====================================================

    def benchmark_llm_inference(self):

        start = time.perf_counter()

        for _, row in self.samples.iterrows():

            _ = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": row["sentence1"]
                    }
                ],
                max_tokens=30
            )

        end = time.perf_counter()

        avg_time_ms = (
            (end - start) / self.sample_size
        ) * 1000

        return round(avg_time_ms, 3)

    # =====================================================
    # RUN
    # =====================================================

    def run(self):

        print("\nRunning benchmarks...\n")

        # =====================================================
        # INDIVIDUAL COMPONENTS
        # =====================================================

        cache_lookup_time = (
            self.benchmark_cache_lookup()
        )

        embedding_time = (
            self.benchmark_embedding_generation()
        )

        semantic_similarity_time = (
            self.benchmark_semantic_similarity()
        )

        feature_time = (
            self.benchmark_feature_computation()
        )

        nn_time = (
            self.benchmark_neural_network()
        )

        transformer_time = (
            self.benchmark_transformer()
        )

        llm_time = (
            self.benchmark_llm_inference()
        )

        # =====================================================
        # TOTAL WORKFLOWS
        # =====================================================

        semantic_total = (
            cache_lookup_time +
            embedding_time +
            semantic_similarity_time
        )

        framework_total = (
            semantic_total +
            feature_time +
            nn_time +
            transformer_time
        )

        # =====================================================
        # RELATIVE COST BASELINE
        # =====================================================

        baseline = cache_lookup_time

        # =====================================================
        # RESULTS TABLE
        # =====================================================

        results = pd.DataFrame([

            {
                "Workflow / Component":
                    "TOTAL SEMANTIC CACHE RETRIEVAL",
                "Avg Time (ms)":
                    round(semantic_total, 3),
                "Relative Cost":
                    f"{semantic_total / baseline:.2f}×"
            },

            {
                "Workflow / Component":
                    "Cache lookup",
                "Avg Time (ms)":
                    cache_lookup_time,
                "Relative Cost":
                    f"{cache_lookup_time / baseline:.2f}×"
            },

            {
                "Workflow / Component":
                    "Embedding generation",
                "Avg Time (ms)":
                    embedding_time,
                "Relative Cost":
                    f"{embedding_time / baseline:.2f}×"
            },

            {
                "Workflow / Component":
                    "Semantic similarity",
                "Avg Time (ms)":
                    semantic_similarity_time,
                "Relative Cost":
                    f"{semantic_similarity_time / baseline:.2f}×"
            },

            {
                "Workflow / Component":
                    "TOTAL FRAMEWORK PIPELINE",
                "Avg Time (ms)":
                    round(framework_total, 3),
                "Relative Cost":
                    f"{framework_total / baseline:.2f}×"
            },

            {
                "Workflow / Component":
                    "Semantic retrieval",
                "Avg Time (ms)":
                    round(semantic_total, 3),
                "Relative Cost":
                    f"{semantic_total / baseline:.2f}×"
            },

            {
                "Workflow / Component":
                    "Feature computation",
                "Avg Time (ms)":
                    feature_time,
                "Relative Cost":
                    f"{feature_time / baseline:.2f}×"
            },

            {
                "Workflow / Component":
                    "Neural drift detection",
                "Avg Time (ms)":
                    nn_time,
                "Relative Cost":
                    f"{nn_time / baseline:.2f}×"
            },

            {
                "Workflow / Component":
                    "Transformer fallback",
                "Avg Time (ms)":
                    transformer_time,
                "Relative Cost":
                    f"{transformer_time / baseline:.2f}×"
            },

            {
                "Workflow / Component":
                    "TOTAL LLM INFERENCE",
                "Avg Time (ms)":
                    llm_time,
                "Relative Cost":
                    f"{llm_time / baseline:.2f}×"
            }

        ])

        print("\n=== Computational Cost Analysis ===\n")

        print(results.to_string(index=False))


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    ROOT_DIR = Path(__file__).resolve().parents[2]

    benchmark = FrameworkComputationalBenchmark(
        nn_model_path=ROOT_DIR / "models" / "drift_model.pt",
        transformer_model_path=ROOT_DIR / "models" / "transformer_model.pt",
        pipeline_path=ROOT_DIR / "models" / "drift_feature_pipeline.pkl",
        sample_size=20
    )

    benchmark.run()