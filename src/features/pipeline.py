import joblib
import os
from sklearn.preprocessing import StandardScaler

from src.features.builder import FeatureBuilder
from src.features.selector import FeatureSelector

class DriftFeaturePipeline:

    def __init__(self):
        self.fb = FeatureBuilder()
        self.scaler = StandardScaler(with_mean=False)
        self.selector = FeatureSelector(C=0.05)

        self.is_fitted = False

    def fit(self, train_df, val_df):
        print("\n===== FITTING FEATURE PIPELINE =====")

        self.fb.fit(train_df)

        X_train, y_train = self.fb.build(train_df)
        X_val, y_val = self.fb.build(val_df)

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        # Feature selection
        self.selector.fit(X_train, y_train, X_val, y_val)

        print("Selected features:", len(self.selector.selected_idx))

        self.is_fitted = True

    def transform(self, df):
        assert self.is_fitted, "DriftFeaturePipeline NOT fitted!"

        X, y = self.fb.build(df)

        X = self.scaler.transform(X)
        X = self.selector.transform(X)

        return X, y

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)