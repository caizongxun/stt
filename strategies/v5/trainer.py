"""
V5 Trainer
V5訓練器
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
    """
    V5訓練器 - 極簡高效
    """
    
    def __init__(self, config: V5Config):
        self.config = config
        self.models = []
        self.feature_names = []
    
    def train(self, df: pd.DataFrame) -> dict:
        """
        完整訓練流程
        """
        print("\n" + "="*60)
        print("V5 TRAINING - Pure ML Price Prediction")
        print("="*60)
        
        # 1. 特徵工程
        feature_engine = V5FeatureEngine(self.config)
        df = feature_engine.generate(df)
        
        # 2. 標籤生成
        label_gen = V5LabelGenerator(self.config)
        df = label_gen.generate(df)
        
        # 3. 準備數據
        self.feature_names = feature_engine.get_feature_names(df)
        X, y, y_direction = self._prepare_data(df)
        
        # 4. 分割數據
        X_train, X_val, X_oos, y_train, y_val, y_oos = self._split_data(X, y)
        
        print(f"\n[Data Split]")
        print(f"  Train: {len(X_train)}")
        print(f"  Val: {len(X_val)}")
        print(f"  OOS: {len(X_oos)}")
        
        # 5. 訓練集成模型
        self.models = self._train_ensemble(X_train, y_train, X_val, y_val)
        
        # 6. 評估
        val_metrics = self._evaluate(self.models, X_val, y_val, "Validation")
        oos_metrics = self._evaluate(self.models, X_oos, y_oos, "OOS")
        
        # 7. 特徵重要性
        feature_importance = self._get_feature_importance()
        
        # 8. 保存模型
        model_path = self._save_models()
        
        results = {
            'val_metrics': val_metrics,
            'oos_metrics': oos_metrics,
            'feature_importance': feature_importance,
            'model_path': model_path,
            'feature_count': len(self.feature_names)
        }
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        
        return results
    
    def _prepare_data(self, df):
        """準備訓練數據"""
        X = df[self.feature_names].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())
        
        y = df['label_binary'].copy()
        y_direction = df['signal_direction'].copy()
        
        valid = y.notna()
        X = X[valid]
        y = y[valid]
        y_direction = y_direction[valid]
        
        return X, y, y_direction
    
    def _split_data(self, X, y):
        """分割數據"""
        n = len(X)
        train_end = int(n * self.config.train_size)
        val_end = int(n * (self.config.train_size + self.config.val_size))
        
        X_train = X.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        X_oos = X.iloc[val_end:]
        
        y_train = y.iloc[:train_end]
        y_val = y.iloc[train_end:val_end]
        y_oos = y.iloc[val_end:]
        
        return X_train, X_val, X_oos, y_train, y_val, y_oos
    
    def _train_ensemble(self, X_train, y_train, X_val, y_val):
        """訓練集成模型"""
        print(f"\n[Training {self.config.ensemble_models} models]")
        
        models = []
        
        for i in range(self.config.ensemble_models):
            # 每個模型用不同子集
            sample_idx = np.random.choice(len(X_train), int(len(X_train) * 0.9), replace=False)
            X_sub = X_train.iloc[sample_idx]
            y_sub = y_train.iloc[sample_idx]
            
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
            print(f"  Model {i+1}/{self.config.ensemble_models} trained")
        
        return models
    
    def _evaluate(self, models, X, y, name):
        """評估模型"""
        # 集成預測
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
        
        print(f"\n[{name} Metrics]")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        
        return metrics
    
    def _get_feature_importance(self):
        """獲取特徵重要性"""
        importances = np.mean([m.feature_importances_ for m in self.models], axis=0)
        feature_imp = list(zip(self.feature_names, importances))
        feature_imp.sort(key=lambda x: x[1], reverse=True)
        return feature_imp[:20]
    
    def _save_models(self):
        """保存模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path(f"models/{self.config.symbol}_{self.config.timeframe}_v5_{timestamp}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        for i, model in enumerate(self.models):
            joblib.dump(model, model_dir / f"model_{i}.pkl")
        
        joblib.dump(self.config.to_dict(), model_dir / "config.pkl")
        joblib.dump(self.feature_names, model_dir / "features.pkl")
        
        print(f"\n[Models saved to: {model_dir}]")
        return str(model_dir)
