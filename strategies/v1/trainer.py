"""
V1 Trainer - Optimized
V1訓練模組 - 優化版
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import json
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.feature_names = []
    
    def train(self, df: pd.DataFrame) -> dict:
        df = self._engineer_features(df)
        df = self._generate_labels(df)
        X, y = self._prepare_data(df)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size=self.config.train_size, shuffle=False
        )
        
        X_train_df = pd.DataFrame(X_train, columns=self.feature_names)
        X_val_df = pd.DataFrame(X_val, columns=self.feature_names)
        
        # 添加正則化參數
        self.model = lgb.LGBMClassifier(
            num_leaves=self.config.num_leaves,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            min_child_samples=self.config.min_child_samples,
            objective='multiclass',
            num_class=3,
            class_weight=self.config.class_weights if self.config.use_class_weight else None,
            reg_alpha=0.1,  # L1正則化
            reg_lambda=0.1,  # L2正則化
            verbosity=-1,
            random_state=42
        )
        
        self.model.fit(X_train_df, y_train, eval_set=[(X_val_df, y_val)], eval_metric='multi_logloss')
        
        train_metrics = self._evaluate(X_train_df, y_train, "train")
        val_metrics = self._evaluate(X_val_df, y_val, "validation")
        feature_importance = self._get_feature_importance()
        train_distribution = self._get_label_distribution(y_train)
        val_distribution = self._get_label_distribution(y_val)
        model_dir = self._save_model()
        
        results = {
            "model_info": {
                "symbol": self.config.symbol,
                "timeframe": self.config.timeframe,
                "model_type": "LightGBM",
                "model_path": str(model_dir),
                "train_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            "data_info": {
                "total_samples": len(X),
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "features_count": len(self.feature_names),
                "train_distribution": train_distribution,
                "val_distribution": val_distribution
            },
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "feature_importance": feature_importance,
            "hyperparameters": self.config.to_dict(),
            "optimization_suggestions": self._generate_suggestions(train_metrics, val_metrics)
        }
        
        self._save_results(results, model_dir)
        return results
    
    def _generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        future_return = df['close'].shift(-self.config.label_periods) / df['close'] - 1
        
        df['label'] = 0
        df.loc[future_return > self.config.label_threshold_long, 'label'] = 1
        df.loc[future_return < self.config.label_threshold_short, 'label'] = 2
        
        return df.dropna()
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 基礎价格特徵
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 移动平均和波动性
        for period in self.config.lookback_periods:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'std_{period}'] = df['close'].rolling(period).std()
            df[f'momentum_{period}'] = df['close'].pct_change(period)
            
            # 价格相对位置
            df[f'price_position_{period}'] = (df['close'] - df['close'].rolling(period).min()) / \
                                              (df['close'].rolling(period).max() - df['close'].rolling(period).min())
        
        # 成交量特徵
        if self.config.use_volume_features:
            df['volume_change'] = df['volume'].pct_change()
            for period in [5, 10, 20]:
                df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
                df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
        
        return df.dropna()
    
    def _prepare_data(self, df: pd.DataFrame):
        exclude_cols = ['open_time', 'close_time', 'label', 'open', 'high', 'low', 'close', 'volume',
                       'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                       'taker_buy_quote_asset_volume', 'ignore']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_names = feature_cols
        X = df[feature_cols].values
        y = df['label'].values
        return X, y
    
    def _evaluate(self, X, y_true, dataset_name: str) -> dict:
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        try:
            auc_macro = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            auc_per_class = roc_auc_score(y_true, y_proba, multi_class='ovr', average=None)
        except:
            auc_macro = 0.0
            auc_per_class = [0.0, 0.0, 0.0]
        
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            "accuracy": float(accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "auc_macro": float(auc_macro),
            "per_class_metrics": {
                "class_0_hold": {
                    "precision": float(precision_per_class[0]),
                    "recall": float(recall_per_class[0]),
                    "f1": float(f1_per_class[0]),
                    "auc": float(auc_per_class[0])
                },
                "class_1_long": {
                    "precision": float(precision_per_class[1]),
                    "recall": float(recall_per_class[1]),
                    "f1": float(f1_per_class[1]),
                    "auc": float(auc_per_class[1])
                },
                "class_2_short": {
                    "precision": float(precision_per_class[2]),
                    "recall": float(recall_per_class[2]),
                    "f1": float(f1_per_class[2]),
                    "auc": float(auc_per_class[2])
                }
            },
            "confusion_matrix": cm.tolist()
        }
    
    def _get_feature_importance(self, top_n: int = 15) -> dict:
        importance = self.model.feature_importances_
        feature_imp = pd.DataFrame({'feature': self.feature_names, 'importance': importance}).sort_values('importance', ascending=False)
        return {"top_features": feature_imp.head(top_n).to_dict('records'), "total_features": len(self.feature_names)}
    
    def _get_label_distribution(self, y) -> dict:
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        return {
            "class_0_hold": {"count": int(counts[0]) if 0 in unique else 0, "percentage": float(counts[0] / total * 100) if 0 in unique else 0.0},
            "class_1_long": {"count": int(counts[1]) if 1 in unique else 0, "percentage": float(counts[1] / total * 100) if 1 in unique else 0.0},
            "class_2_short": {"count": int(counts[2]) if 2 in unique else 0, "percentage": float(counts[2] / total * 100) if 2 in unique else 0.0}
        }
    
    def _generate_suggestions(self, train_metrics: dict, val_metrics: dict) -> list:
        suggestions = []
        train_acc = train_metrics['accuracy']
        val_acc = val_metrics['accuracy']
        
        if train_acc - val_acc > 0.15:
            suggestions.append({"issue": "overfitting", "description": f"訓練集({train_acc:.3f})遠高於驗證集({val_acc:.3f})", "recommendation": "減少max_depth或增加min_child_samples"})
        if val_acc < 0.5:
            suggestions.append({"issue": "low_accuracy", "description": f"驗證集准確率過低({val_acc:.3f})", "recommendation": "優化特徵或調整標籤閘值"})
        if val_metrics['auc_macro'] < 0.7:
            suggestions.append({"issue": "low_auc", "description": f"AUC過低({val_metrics['auc_macro']:.3f})", "recommendation": "優化特徵工程"})
        
        # 檢查各類別平衡
        for class_name, metrics in val_metrics['per_class_metrics'].items():
            if metrics['recall'] < 0.2:
                suggestions.append({"issue": "class_imbalance", "description": f"{class_name}召回率過低({metrics['recall']:.3f})", "recommendation": "調整權重或閘值"})
            if metrics['recall'] > 0.8 and metrics['precision'] < 0.3:
                suggestions.append({"issue": "overprediction", "description": f"{class_name}過度預測(召回{metrics['recall']:.3f}/精確{metrics['precision']:.3f})", "recommendation": "降低權重或提高閘值"})
        
        if not suggestions:
            suggestions.append({"issue": "none", "description": "模型表現良好", "recommendation": "可以進行回測"})
        return suggestions
    
    def _save_model(self) -> Path:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"{self.config.symbol}_{self.config.timeframe}_v1_{timestamp}"
        model_dir = Path('models') / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_dir / 'model.pkl')
        joblib.dump(self.config.to_dict(), model_dir / 'config.pkl')
        joblib.dump(self.feature_names, model_dir / 'features.pkl')
        return model_dir
    
    def _save_results(self, results: dict, model_dir: Path):
        results_path = model_dir / 'training_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
