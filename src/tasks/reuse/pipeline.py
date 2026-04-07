import torch
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score

from core.features import FeatureBuilder
from core.feature_selector import FeatureSelector

from core.model import ModelBuilder
from tasks.reuse.trainer import Trainer
from tasks.reuse.tuner import OptunaTuner


class ReusePipeline:

    def __init__(self):
        self.fb = FeatureBuilder()
        self.scaler = StandardScaler(with_mean=False)
        self.selector = FeatureSelector(C=0.05)

        self.model = None
        self.threshold = 0.6
        self.calibrator = None

    def fit(self, train_df, val_df):

        print("\n===== REUSE PIPELINE =====")

        # ===== FEATURES =====
        self.fb.fit(train_df)

        X_train, y_train = self.fb.build(train_df)
        X_val, y_val = self.fb.build(val_df)

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        self.selector.fit(X_train, y_train, X_val, y_val)

        X_train = self.selector.transform(X_train)
        X_val = self.selector.transform(X_val)

        X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
        X_val = torch.tensor(X_val.toarray(), dtype=torch.float32)

        y_train = torch.tensor(y_train)
        y_val = torch.tensor(y_val)

        # ===== TUNING =====
        tuner = OptunaTuner(X_train, y_train, X_val, y_val)
        params = tuner.tune()

        # ===== MODEL =====
        self.model = ModelBuilder.build(
            X_train.shape[1],
            params["n_layers"],
            params["hidden"],
            params["dropout"]
        )

        trainer = Trainer(self.model, params)
        self.model = trainer.train(X_train, y_train, X_val, y_val)

        # fallback threshold
        self.threshold = trainer.best_threshold

        # =========================
        # 🔥 КАЛИБРОВКА (logits)
        # =========================
        with torch.no_grad():
            logits = self.model(X_val).numpy()

        self.calibrator = LogisticRegression()
        self.calibrator.fit(logits, y_val.numpy())

        # =========================
        # 🔥 ПОДБОР THRESHOLD
        # =========================
        probs = self.calibrator.predict_proba(logits)[:, 1]

        best_t = self.threshold
        best_score = 0

        for t in np.linspace(0.5, 0.95, 120):

            preds = (probs > t).astype(int)

            precision = precision_score(y_val.numpy(), preds, zero_division=0)
            f1 = f1_score(y_val.numpy(), preds)

            # ❗ фильтр по precision
            if precision < 0.7:
                continue

            if f1 > best_score:
                best_score = f1
                best_t = t

        self.threshold = best_t

        print(f"[CALIBRATION] New threshold: {self.threshold:.3f}")

    def predict(self, df):

        X, _ = self.fb.build(df)

        X = self.scaler.transform(X)
        X = self.selector.transform(X)

        X = torch.tensor(X.toarray(), dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(X).numpy()

        probs = self.calibrator.predict_proba(logits)[:, 1]

        preds = (probs > self.threshold).astype(int)

        return preds, probs