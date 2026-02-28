"""
V6 Trainer - Reversal Prediction
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from .config import V6Config
from .features import V6FeatureEngineer
from .labels import V6LabelGenerator


class V6Trainer:
    def __init__(self, config: V6Config):
        self.config = config
        self.long_model = None
        self.short_model = None
        self.feature_names = []

    def train(self, df: pd.DataFrame) -> dict:
        print("\n" + "=" * 50)
        print("V6 TRAINING - Reversal Prediction")
        print("=" * 50)

        feature_engine = V6FeatureEngineer(self.config)
        df = feature_engine.generate(df)

        label_gen = V6LabelGenerator(self.config)
        df = label_gen.generate(df)

        self.feature_names = feature_engine.get_feature_names(df)
        X = self._prepare_features(df)

        y_long = (df["label"] == 1).astype(int)
        y_short = (df["label"] == -1).astype(int)
        valid = y_long.notna() & y_short.notna()
        X = X[valid]
        y_long = y_long[valid]
        y_short = y_short[valid]

        (
            X_train,
            X_val,
            y_long_train,
            y_long_val,
            y_short_train,
            y_short_val,
        ) = self._split_data(X, y_long, y_short)

        print(f"[Data] Train: {len(X_train)}, Val: {len(X_val)}")
        print(
            f"[Pos Rate] Long: {y_long_train.mean()*100:.2f}%, Short: {y_short_train.mean()*100:.2f}%"
        )

        self.long_model = self._train_model(
            X_train, y_long_train, X_val, y_long_val, "LONG"
        )
        self.short_model = self._train_model(
            X_train, y_short_train, X_val, y_short_val, "SHORT"
        )

        long_val = self._evaluate(self.long_model, X_val, y_long_val)
        short_val = self._evaluate(self.short_model, X_val, y_short_val)

        model_path = self._save_models()

        return {
            "long_val_metrics": long_val,
            "short_val_metrics": short_val,
            "feature_count": len(self.feature_names),
            "model_path": model_path,
        }

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_names].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())
        return X

    def _split_data(self, X, y_long, y_short):
        n = len(X)
        train_end = int(n * self.config.train_size)
        val_end = int(n * (self.config.train_size + self.config.val_size))

        return (
            X.iloc[:train_end],
            X.iloc[train_end:val_end],
            y_long.iloc[:train_end],
            y_long.iloc[train_end:val_end],
            y_short.iloc[:train_end],
            y_short.iloc[train_end:val_end],
        )

    def _train_model(self, X_train, y_train, X_val, y_val, name: str):
        pos = y_train.sum()
        if pos == 0:
            print(f"[{name}] 無正樣本，略過")
            return None

        neg = len(y_train) - pos
        class_weight = {0: 1.0, 1: max(1.0, neg / (pos + 1e-6))}

        model = lgb.LGBMClassifier(
            num_leaves=self.config.num_leaves,
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            class_weight=class_weight,
            objective="binary",
            random_state=42,
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        print(f"[{name}] 訓練完成")
        return model

    def _evaluate(self, model, X, y):
        if model is None or len(X) == 0:
            return {"accuracy": 0, "precision": 0, "recall": 0, "auc": 0}

        proba = model.predict_proba(X)[:, 1]
        y_pred = (proba >= 0.5).astype(int)

        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "auc": roc_auc_score(y, proba) if y.nunique() > 1 else 0.5,
        }

    def _save_models(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path(
            f"models/{self.config.symbol}_{self.config.timeframe}_v6_{timestamp}"
        )
        model_dir.mkdir(parents=True, exist_ok=True)

        if self.long_model:
            joblib.dump(self.long_model, model_dir / "long_model.pkl")
        if self.short_model:
            joblib.dump(self.short_model, model_dir / "short_model.pkl")

        joblib.dump(self.config.to_dict(), model_dir / "config.pkl")
        joblib.dump(self.feature_names, model_dir / "features.pkl")

        print(f"[Saved] {model_dir}")
        return str(model_dir)
