"""
V1 Trainer
V1訓練模組
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import json

class Trainer:
    """
    V1訓練器
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.feature_names = []
    
    def train(self, df: pd.DataFrame) -> dict:
        """
        訓練模型
        
        Args:
            df: K線數據
        
        Returns:
            dict: 詳細訓練結果
        """
        # 1. 特徵工程
        df = self._engineer_features(df)
        
        # 2. 生成標籤
        df = self._generate_labels(df)
        
        # 3. 準備訓練數據
        X, y = self._prepare_data(df)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size=self.config.train_size, shuffle=False
        )
        
        # 4. 訓練模型
        self.model = lgb.LGBMClassifier(
            num_leaves=self.config.num_leaves,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            objective='multiclass',
            num_class=3,
            verbosity=-1,
            random_state=42
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='multi_logloss'
        )
        
        # 5. 詳細評估
        train_metrics = self._evaluate(X_train, y_train, "train")
        val_metrics = self._evaluate(X_val, y_val, "validation")
        
        # 6. 特徵重要性
        feature_importance = self._get_feature_importance()
        
        # 7. 標籤分佈
        train_distribution = self._get_label_distribution(y_train)
        val_distribution = self._get_label_distribution(y_val)
        
        # 8. 保存
        model_dir = self._save_model()
        
        # 9. 組裝結果
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
            "hyperparameters": {
                "num_leaves": self.config.num_leaves,
                "max_depth": self.config.max_depth,
                "learning_rate": self.config.learning_rate,
                "n_estimators": self.config.n_estimators,
                "train_size": self.config.train_size
            },
            "optimization_suggestions": self._generate_suggestions(train_metrics, val_metrics)
        }
        
        # 保存結果到JSON
        self._save_results(results, model_dir)
        
        return results
    
    def _evaluate(self, X, y_true, dataset_name: str) -> dict:
        """
        詳細評估指標
        """
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        
        # 基礎指標
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # 每類指標
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # AUC (One-vs-Rest)
        try:
            auc_macro = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            auc_per_class = roc_auc_score(y_true, y_proba, multi_class='ovr', average=None)
        except:
            auc_macro = 0.0
            auc_per_class = [0.0, 0.0, 0.0]
        
        # 混淆矩陣
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
    
    def _get_feature_importance(self, top_n: int = 10) -> dict:
        """獲取特徵重要性"""
        importance = self.model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return {
            "top_features": feature_imp.head(top_n).to_dict('records'),
            "total_features": len(self.feature_names)
        }
    
    def _get_label_distribution(self, y) -> dict:
        """獲取標籤分佈"""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        return {
            "class_0_hold": {
                "count": int(counts[0]) if 0 in unique else 0,
                "percentage": float(counts[0] / total * 100) if 0 in unique else 0.0
            },
            "class_1_long": {
                "count": int(counts[1]) if 1 in unique else 0,
                "percentage": float(counts[1] / total * 100) if 1 in unique else 0.0
            },
            "class_2_short": {
                "count": int(counts[2]) if 2 in unique else 0,
                "percentage": float(counts[2] / total * 100) if 2 in unique else 0.0
            }
        }
    
    def _generate_suggestions(self, train_metrics: dict, val_metrics: dict) -> list:
        """
        生成優化建議
        """
        suggestions = []
        
        # 1. 檢查過擬合
        train_acc = train_metrics['accuracy']
        val_acc = val_metrics['accuracy']
        
        if train_acc - val_acc > 0.1:
            suggestions.append({
                "issue": "overfitting",
                "description": f"訓練集准確率({train_acc:.3f})遠高於驗證集({val_acc:.3f})",
                "recommendation": "建議: 減少max_depth或num_leaves, 或增加正則化"
            })
        
        # 2. 檢查欠擬合
        if val_acc < 0.6:
            suggestions.append({
                "issue": "underfitting",
                "description": f"驗證集准確率過低({val_acc:.3f})",
                "recommendation": "建議: 增加模型複雜度或增加特徵"
            })
        
        # 3. 檢查AUC
        val_auc = val_metrics['auc_macro']
        if val_auc < 0.7:
            suggestions.append({
                "issue": "low_auc",
                "description": f"AUC過低({val_auc:.3f})",
                "recommendation": "建議: 優化特徵工程或調整標籤生成逻輯"
            })
        
        # 4. 檢查類別不平衡
        for class_name, metrics in val_metrics['per_class_metrics'].items():
            if metrics['recall'] < 0.3:
                suggestions.append({
                    "issue": "class_imbalance",
                    "description": f"{class_name}召回率過低({metrics['recall']:.3f})",
                    "recommendation": "建議: 使用類別權重或SMOTE採樣"
                })
        
        if not suggestions:
            suggestions.append({
                "issue": "none",
                "description": "模型表現良好",
                "recommendation": "可以進行回測"
            })
        
        return suggestions
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徵工程"""
        df = df.copy()
        
        # 基礎技術指標
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        for period in self.config.lookback_periods:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'std_{period}'] = df['close'].rolling(period).std()
            df[f'momentum_{period}'] = df['close'].pct_change(period)
        
        return df.dropna()
    
    def _generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成標籤"""
        df = df.copy()
        
        # 簡單標籤: 未來5期報酬
        future_return = df['close'].shift(-5) / df['close'] - 1
        
        # 0: 不交易, 1: 做多, 2: 做空
        df['label'] = 0
        df.loc[future_return > 0.01, 'label'] = 1  # 做多
        df.loc[future_return < -0.01, 'label'] = 2  # 做空
        
        return df.dropna()
    
    def _prepare_data(self, df: pd.DataFrame):
        """準備訓練數據"""
        exclude_cols = ['open_time', 'close_time', 'label', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        self.feature_names = feature_cols
        
        X = df[feature_cols].values
        y = df['label'].values
        
        return X, y
    
    def _save_model(self) -> Path:
        """保存模型"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"{self.config.symbol}_{self.config.timeframe}_v1_{timestamp}"
        model_dir = Path('models') / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        joblib.dump(self.model, model_dir / 'model.pkl')
        
        # 保存配置
        joblib.dump(self.config.to_dict(), model_dir / 'config.pkl')
        
        # 保存特徵名
        joblib.dump(self.feature_names, model_dir / 'features.pkl')
        
        return model_dir
    
    def _save_results(self, results: dict, model_dir: Path):
        """保存結果到JSON"""
        results_path = model_dir / 'training_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
