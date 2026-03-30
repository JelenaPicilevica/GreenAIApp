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

    loader = DatasetLoader()
    train_df, val_df, test_df = loader.load()

    pipeline = DriftPipeline()
    pipeline.fit(train_df, val_df)

    probs = pipeline.predict_proba(test_df)
    y_true = test_df["drift"].values

    nn_f1, nn_acc, auc_nn = Evaluator.evaluate("NN", y_true, probs)

    transformer = TransformerFallback()

    tr_probs = []
    for i in range(len(test_df)):
        s1 = test_df["sentence1"].iloc[i]
        s2 = test_df["sentence2"].iloc[i]
        _, p = transformer.is_drift(s1, s2)
        tr_probs.append(p)

    tr_probs = np.array(tr_probs)

    tr_f1, tr_acc, auc_tr = Evaluator.evaluate("Transformer", y_true, tr_probs)

    router = DriftRouter(transformer)
    final_probs = router.route(probs, test_df)

    hybrid_f1, hybrid_acc, auc_hybrid = Evaluator.evaluate("Hybrid", y_true, final_probs)

    print("\n===== SUMMARY =====")
    print(f"NN AUC: {auc_nn:.3f}")
    print(f"Transformer AUC: {auc_tr:.3f}")
    print(f"Hybrid AUC: {auc_hybrid:.3f}")

    # =====================
    # DEBUG
    # =====================
    Evaluator.debug_hybrid(y_true, probs, final_probs)


if __name__ == "__main__":
    main()