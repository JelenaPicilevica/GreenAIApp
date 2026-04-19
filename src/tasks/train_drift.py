import os
import numpy as np
import torch

from src.data.dataset import DatasetLoader
from src.model.transformer_fallback import TransformerFallback

from src.pipeline.drift_pipeline import DriftPipeline
from src.pipeline.router import DriftRouter
from src.evaluation.evaluator import Evaluator

from src.features.pipeline import DriftFeaturePipeline


def main():
    print("\n🚀 START PIPELINE\n")

    # =====================
    # DATA
    # =====================
    loader = DatasetLoader()
    train_df, val_df, test_df = loader.load()

    # =====================
    # FEATURE PIPELINE
    # =====================
    features = DriftFeaturePipeline()
    features.fit(train_df, val_df)

    # =====================
    # NN PIPELINE
    # =====================
    pipeline = DriftPipeline(features)
    pipeline.fit(train_df, val_df)

    probs = pipeline.predict_proba(test_df)
    y_true = test_df["drift"].values

    results = []

    # =====================
    # NN
    # =====================
    results.append(Evaluator.evaluate_full("NN", y_true, probs))

    # =====================
    # TRANSFORMER
    # =====================
    transformer = TransformerFallback()

    tr_probs = []
    for i in range(len(test_df)):
        s1 = test_df["sentence1"].iloc[i]
        s2 = test_df["sentence2"].iloc[i]
        _, p = transformer.is_drift(s1, s2)
        tr_probs.append(p)

    tr_probs = np.array(tr_probs)

    results.append(Evaluator.evaluate_full("Transformer", y_true, tr_probs))

    # =====================
    # HYBRID
    # =====================
    router = DriftRouter(transformer)
    final_probs = router.route(probs, test_df)

    results.append(Evaluator.evaluate_full("Hybrid", y_true, final_probs))

    # =====================
    # PLOTS
    # =====================
    models = {
        "Neural Network": probs,
        "Transformer": tr_probs,
        "Hybrid": final_probs
    }

    Evaluator.plot_all_confusion(y_true, models)
    Evaluator.plot_roc_all(y_true, models)

    # =====================
    # DEBUG
    # =====================
    Evaluator.debug_hybrid(y_true, probs, final_probs)

    # =====================
    # FINAL TABLE
    # =====================
    Evaluator.print_comparison_table(results)

    # =====================
    # SAVE LOCATION (FIXED)
    # =====================
    BASE_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../..")
    )

    MODELS_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\n Saving to directory:", MODELS_DIR)

    # =====================
    # SAVE FEATURE PIPELINE
    # =====================
    PIPELINE_PATH = os.path.join(MODELS_DIR, "drift_feature_pipeline.pkl")

    print("\n Saving feature pipeline...")
    features.save(PIPELINE_PATH)

    print("Saved to:", PIPELINE_PATH)

    # =====================
    # SAVE MODEL
    # =====================
    MODEL_PATH = os.path.join(MODELS_DIR, "drift_model.pt")

    print("\n Saving drift model...")
    torch.save(pipeline.model, MODEL_PATH)

    print("Saved model to:", MODEL_PATH)

    # =====================
    # VERIFY MODEL
    # =====================
    model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.eval()

    print("Model type:", type(model))

    input_dim = next(model.parameters()).shape[1]
    test_input = torch.randn(1, input_dim)

    with torch.no_grad():
        out = model(test_input)

    print("Model output shape:", out.shape)
    print("✅ Model works correctly")


if __name__ == "__main__":
    main()