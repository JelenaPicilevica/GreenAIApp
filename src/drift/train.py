import numpy as np

from drift.neuralNetwork.dataset import DatasetLoader
from drift.transformer.transformer_fallback import TransformerFallback

from drift.pipeline.drift_pipeline import DriftPipeline
from drift.pipeline.drift_router import DriftRouter
from drift.pipeline.evaluator import Evaluator


def main():
    """
    Entry point for training and evaluating
    neural network drift model and comparing it
    with transformer and hybrid approach.
    """

    print("\n🚀 START PIPELINE\n")

    # =====================
    # DATA
    # =====================
    loader = DatasetLoader()
    train_df, val_df, test_df = loader.load()

    # =====================
    # NN PIPELINE
    # =====================
    pipeline = DriftPipeline()
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


if __name__ == "__main__":
    main()