"""
V2 Trainer
V2訓練模組
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .label_generator import LabelGenerator
from .feature_engineer import FeatureEngineer

class Trainer:
    def __init__(self, config):
        self.config = config
        self.label_generator = LabelGenerator(config)
        self.feature_engineer = FeatureEngineer(config)
    
    def train(self, df: pd.DataFrame) -> dict:
        """
        完整訓練流程
        """
        # 1. 生成標籤
        df = self.label_generator.generate(df)
        label_stats = self.label_generator.get_statistics(df)
        
        # 2. 生成特徵
        df, feature_names = self.feature_engineer.engineer(df)
        
        # 3. 只使用有BB觸碰的樣本
        df_train = df[(df['touch_upper'] | df['touch_lower'])].copy()
        df_train = df_train.dropna()
        
        if len(df_train) < 100:
            raise ValueError(f"訓練樣本太少: {len(df_train)}, 需要至少100筆")
        
        # 4. 準備訓練數據
        X = df_train[feature_names]
        y = df_train['label']
        
        # 5. 分割數據
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=1-self.config.train_size, shuffle=False
        )
        
        # 6. 訓練模型
        model = self._train_model(X_train, y_train, X_val, y_val)
        
        # 7. 評估模型
        train_metrics = self._evaluate(model, X_train, y_train, feature_names)
        val_metrics = self._evaluate(model, X_val, y_val, feature_names)
        
        # 8. 保存模型
        model_path = self._save_model(model, feature_names)
        
        # 9. 返回結果
        results = {
            'model_info': {
                'symbol': self.config.symbol,
                'timeframe': self.config.timeframe,
                'model_type': self.config.model_type,
                'model_path': str(model_path),
                'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'label_statistics': label_stats,
            'data_info': {
                'total_samples': len(df),
                'bb_touch_samples': len(df_train),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'features_count': len(feature_names),
                'class_distribution': {
                    'negative': int((y == 0).sum()),
                    'positive': int((y == 1).sum()),
                    'positive_rate': float((y == 1).sum() / len(y) * 100)
                }
            },
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        return results
    
    def _train_model(self, X_train, y_train, X_val, y_val):
        """
        訓練LightGBM模型
        """
        # 修正class_weight的key類型
        if self.config.use_class_weight:
            # 確保y_train中實際有哪些類別
            unique_classes = np.unique(y_train)
            # 將config中的weight轉換為字典,key使用numpy類型
            class_weight = {int(k): v for k, v in self.config.class_weights.items() if k in unique_classes}
        else:
            class_weight = None
        
        model = lgb.LGBMClassifier(
            objective='binary',
            num_leaves=self.config.num_leaves,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            min_child_samples=self.config.min_child_samples,
            class_weight=class_weight,
            random_state=42,
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc'
        )
        
        return model
    
    def _evaluate(self, model, X, y, feature_names) -> dict:
        """
        評估模型
        """
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred, zero_division=0)),
            'recall': float(recall_score(y, y_pred, zero_division=0)),
            'f1': float(f1_score(y, y_pred, zero_division=0)),
            'auc': float(roc_auc_score(y, y_proba)) if len(np.unique(y)) > 1 else 0.0,
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
        
        return metrics
    
    def _save_model(self, model, feature_names) -> Path:
        """
        保存模型
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"{self.config.symbol}_{self.config.timeframe}_v2_{timestamp}"
        model_dir = Path('models') / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, model_dir / 'model.pkl')
        joblib.dump(self.config.to_dict(), model_dir / 'config.pkl')
        joblib.dump(feature_names, model_dir / 'features.pkl')
        
        return model_dir
