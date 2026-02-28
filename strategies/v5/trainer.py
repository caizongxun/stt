"""
V5 Trainer - Dual Model System
V5訓練器 - 做多/做空分離
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import joblib
from pathlib import Path
from datetime import datetime

from .config import V5Config
from .features import V5FeatureEngine
from .labels import V5LabelGenerator

class V5Trainer:
    def __init__(self, config: V5Config):
        self.config = config
        self.long_models = []
        self.short_models = []
        self.feature_names = []
    
    def train(self, df: pd.DataFrame) -> dict:
        print("\n" + "="*60)
        print("V5 TRAINING - Dual Model System")
        print("="*60)
        
        # 1. 特徵工程
        feature_engine = V5FeatureEngine(self.config)
        df = feature_engine.generate(df)
        
        # 2. 標籤生成
        label_gen = V5LabelGenerator(self.config)
        df = label_gen.generate(df)
        
        # 3. 準備數據
        self.feature_names = feature_engine.get_feature_names(df)
        X = self._prepare_features(df)
        
        # 4. 分離做多/做空標籤
        y_long = df['label_long'].copy()
        y_short = df['label_short'].copy()
        
        valid = y_long.notna() & y_short.notna()
        X = X[valid]
        y_long = y_long[valid]
        y_short = y_short[valid]
        
        # 5. 分割數據
        X_train, X_val, X_oos, y_long_train, y_long_val, y_long_oos, y_short_train, y_short_val, y_short_oos = \
            self._split_data(X, y_long, y_short)
        
        print(f"\n[Data Split]")
        print(f"  Train: {len(X_train)}")
        print(f"  Val: {len(X_val)}")
        print(f"  OOS: {len(X_oos)}")
        print(f"  Long opportunities: {y_long_train.sum()}")
        print(f"  Short opportunities: {y_short_train.sum()}")
        
        # 6. 訓練做多模型
        print("\n[Training LONG models]")
        self.long_models = self._train_ensemble(X_train, y_long_train, X_val, y_long_val, "LONG")
        
        # 7. 訓練做空模型
        print("\n[Training SHORT models]")
        self.short_models = self._train_ensemble(X_train, y_short_train, X_val, y_short_val, "SHORT")
        
        # 8. 評估
        long_val = self._evaluate(self.long_models, X_val, y_long_val, "LONG Val")
        long_oos = self._evaluate(self.long_models, X_oos, y_long_oos, "LONG OOS")
        short_val = self._evaluate(self.short_models, X_val, y_short_val, "SHORT Val")
        short_oos = self._evaluate(self.short_models, X_oos, y_short_oos, "SHORT OOS")
        
        # 9. 特徵重要性
        long_importance = self._get_feature_importance(self.long_models, "LONG")
        short_importance = self._get_feature_importance(self.short_models, "SHORT")
        
        # 10. 保存
        model_path = self._save_models()
        
        results = {
            'long_val_metrics': long_val,
            'long_oos_metrics': long_oos,
            'short_val_metrics': short_val,
            'short_oos_metrics': short_oos,
            'long_feature_importance': long_importance,
            'short_feature_importance': short_importance,
            'model_path': model_path,
            'feature_count': len(self.feature_names)
        }
        
        print("\n" + "="*60)
        print("DUAL MODEL TRAINING COMPLETE")
        print("="*60)
        
        return results
    
    def _prepare_features(self, df):
        X = df[self.feature_names].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())
        return X
    
    def _split_data(self, X, y_long, y_short):
        n = len(X)
        train_end = int(n * self.config.train_size)
        val_end = int(n * (self.config.train_size + self.config.val_size))
        
        return (X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:],
                y_long.iloc[:train_end], y_long.iloc[train_end:val_end], y_long.iloc[val_end:],
                y_short.iloc[:train_end], y_short.iloc[train_end:val_end], y_short.iloc[val_end:])
    
    def _train_ensemble(self, X_train, y_train, X_val, y_val, name):
        models = []
        for i in range(self.config.ensemble_models):
            sample_idx = np.random.choice(len(X_train), int(len(X_train) * 0.9), replace=False)
            X_sub, y_sub = X_train.iloc[sample_idx], y_train.iloc[sample_idx]
            
            model = xgb.XGBClassifier(
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                n_estimators=self.config.n_estimators,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                min_child_weight=self.config.min_child_weight,
                gamma=self.config.gamma,
                random_state=42 + i,
                n_jobs=-1
            )
            
            model.fit(X_sub, y_sub, eval_set=[(X_val, y_val)], verbose=False)
            models.append(model)
            print(f"  {name} Model {i+1}/{self.config.ensemble_models}")
        return models
    
    def _evaluate(self, models, X, y, name):
        probas = [m.predict_proba(X)[:, 1] for m in models]
        y_proba = np.mean(probas, axis=0)
        y_pred = (y_proba >= 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'auc': roc_auc_score(y, y_proba)
        }
        
        cm = confusion_matrix(y, y_pred)
        if cm.shape == (2, 2):
            metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
        
        print(f"\n[{name}] AUC:{metrics['auc']:.3f} P:{metrics['precision']:.3f} R:{metrics['recall']:.3f}")
        return metrics
    
    def _get_feature_importance(self, models, name):
        importances = np.mean([m.feature_importances_ for m in models], axis=0)
        feature_imp = list(zip(self.feature_names, importances))
        feature_imp.sort(key=lambda x: x[1], reverse=True)
        return feature_imp[:15]
    
    def _save_models(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path(f"models/{self.config.symbol}_{self.config.timeframe}_v5_{timestamp}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        for i, model in enumerate(self.long_models):
            joblib.dump(model, model_dir / f"long_model_{i}.pkl")
        for i, model in enumerate(self.short_models):
            joblib.dump(model, model_dir / f"short_model_{i}.pkl")
        
        joblib.dump(self.config.to_dict(), model_dir / "config.pkl")
        joblib.dump(self.feature_names, model_dir / "features.pkl")
        
        print(f"\n[Models saved: {model_dir}]")
        return str(model_dir)
