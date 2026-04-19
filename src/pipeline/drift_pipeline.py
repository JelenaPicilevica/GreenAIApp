import torch
from sklearn.linear_model import LogisticRegression

from src.model.drift_model import ModelBuilder
from src.model.trainer import Trainer
from src.model.tuner import OptunaTuner


class DriftPipeline:

    def __init__(self, shared_pipeline):
        self.shared = shared_pipeline
        self.model = None
        self.calibrator = None
        self.params = None

    def fit(self, train_df, val_df):

        print("\n===== DRIFT PIPELINE =====")

        X_train, y_train = self.shared.transform(train_df)
        X_val, y_val = self.shared.transform(val_df)

        print("Drift features shape:", X_train.shape)

        X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
        X_val = torch.tensor(X_val.toarray(), dtype=torch.float32)

        y_train = torch.tensor(y_train)
        y_val = torch.tensor(y_val)

        # ===== TUNING =====
        tuner = OptunaTuner(X_train, y_train, X_val, y_val)
        self.params = tuner.tune()

        # ===== MODEL =====
        self.model = ModelBuilder.build(
            X_train.shape[1],
            self.params["n_layers"],
            self.params["hidden"],
            self.params["dropout"]
        )

        trainer = Trainer(self.model, self.params)
        self.model = trainer.train(X_train, y_train, X_val, y_val)

        # ===== CALIBRATION =====
        with torch.no_grad():
            val_probs = torch.softmax(self.model(X_val), dim=1)[:, 1].numpy()

        self.calibrator = LogisticRegression()
        self.calibrator.fit(val_probs.reshape(-1, 1), y_val.numpy())

    def save(self, path="models/drift_model.pt"):
        print("\n💾 Saving drift model...")

        torch.save({
            "model": self.model.state_dict(),
            "params": self.params
        }, path)

        print("✅ Drift model saved")

    def predict_proba(self, df):

        X, _ = self.shared.transform(df)
        X = torch.tensor(X.toarray(), dtype=torch.float32)

        with torch.no_grad():
            raw_probs = torch.softmax(self.model(X), dim=1)[:, 1].numpy()

        probs = self.calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        return probs