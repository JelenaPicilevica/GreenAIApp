import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.core.feature_selector import FeatureSelector
from src.core.features import FeatureBuilder
from src.core.model import ModelBuilder
from src.tasks.drift.trainer import Trainer
from src.tasks.drift.tuner import OptunaTuner


class DriftPipeline:
    """
    Builds and trains neural network drift model,
    including feature engineering, scaling,
    feature selection, tuning and calibration.
    """

    def __init__(self):
        self.fb = FeatureBuilder()
        self.scaler = StandardScaler(with_mean=False)
        self.selector = FeatureSelector(C=0.05)

        self.model = None
        self.calibrator = None

    def fit(self, train_df, val_df):
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

        tuner = OptunaTuner(X_train, y_train, X_val, y_val)
        params = tuner.tune()

        self.model = ModelBuilder.build(
            X_train.shape[1],
            params["n_layers"],
            params["hidden"],
            params["dropout"]
        )

        trainer = Trainer(self.model, params)
        self.model = trainer.train(X_train, y_train, X_val, y_val)

        with torch.no_grad():
            val_probs = torch.softmax(self.model(X_val), dim=1)[:, 1].numpy()

        self.calibrator = LogisticRegression()
        self.calibrator.fit(val_probs.reshape(-1, 1), y_val.numpy())

    def predict_proba(self, df):
        X, _ = self.fb.build(df)

        print("Features before selection:", X.shape)

        X = self.scaler.transform(X)
        X = self.selector.transform(X)

        print("Features after selection:", X.shape)

        X = torch.tensor(X.toarray(), dtype=torch.float32)

        with torch.no_grad():
            raw_probs = torch.softmax(self.model(X), dim=1)[:, 1].numpy()

        probs = self.calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        return probs