"""
V3 Trainer
V3訓練器 - 集成XGBoost模型
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
from pathlib import Path
from datetime import datetime

from .signal_generators import SignalGenerator
from .label_generator import LabelGenerator
from .feature_engineer import FeatureEngineer
from .oos_validator import OOSValidator

class EnsembleTrainer:
    """集成模型訓練器"""
    
    def __init__(self, config):
        self.config = config
        self.models = []
        self.feature_names = []
    
    def train(self, df: pd.DataFrame) -> dict:
        """
        完整訓練流程
        1. 生成信號和標籤
        2. OOS分割
        3. 特徵工程
        4. 集成訓練
        5. OOS驗證
        """
        print("\n[V3 Ensemble Training]")
        
        # 1. 信號生成
        print("Step 1: Generating signals...")
        signal_gen = SignalGenerator(self.config)
        df = signal_gen.generate_all_signals(df)
        
        # 2. 標籤生成
        print("Step 2: Generating labels...")
        label_gen = LabelGenerator(self.config)
        df = label_gen.generate_labels(df)
        label_stats = label_gen.get_label_statistics(df)
        print(f"  - Positive rate: {label_stats['positive_rate']:.1f}%")
        print(f"  - Long labels: {label_stats['long_labels']}")
        print(f"  - Short labels: {label_stats['short_labels']}")
        
        # 3. OOS分割
        print("Step 3: OOS data split...")
        oos_validator = OOSValidator(self.config)
        data_splits = oos_validator.split_data(df)
        
        validation = oos_validator.validate_no_leakage(
            data_splits['train'],
            data_splits['val'],
            data_splits['oos']
        )
        print(f"  - Train: {data_splits['split_info']['train_bars']} bars")
        print(f"  - Val: {data_splits['split_info']['val_bars']} bars")
        print(f"  - OOS: {data_splits['split_info']['oos_bars']} bars")
        print(f"  - No leakage: {validation['is_valid']}")
        
        # 4. 特徵工程
        print("Step 4: Feature engineering...")
        feat_eng = FeatureEngineer(self.config)
        
        train_df, feature_names = feat_eng.engineer(data_splits['train'])
        val_df, _ = feat_eng.engineer(data_splits['val'])
        oos_df, _ = feat_eng.engineer(data_splits['oos'])
        
        self.feature_names = feature_names
        print(f"  - Features: {len(feature_names)}")
        
        # 5. 集成訓練
        print("Step 5: Training ensemble models...")
        results = self._train_ensemble(
            train_df, val_df, oos_df, feature_names
        )
        
        # 6. 保存模型
        print("Step 6: Saving models...")
        model_path = self._save_models(results)
        print(f"  - Saved to: {model_path}")
        
        # 7. 返回結果
        results['label_statistics'] = label_stats
        results['split_info'] = data_splits['split_info']
        results['validation_check'] = validation
        results['model_path'] = str(model_path)
        
        return results
    
    def _train_ensemble(self, train_df, val_df, oos_df, feature_names) -> dict:
        """
        訓練集成模型
        """
        # 準備數據
        X_train = train_df[feature_names]
        y_train = train_df['label']
        
        X_val = val_df[feature_names]
        y_val = val_df['label']
        
        X_oos = oos_df[feature_names]
        y_oos = oos_df['label']
        
        # 清理數據: 處理inf和nan
        print("  - Cleaning data...")
        X_train = self._clean_data(X_train)
        X_val = self._clean_data(X_val)
        X_oos = self._clean_data(X_oos)
        
        # 移除NaN行
        train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
        val_mask = ~(X_val.isna().any(axis=1) | y_val.isna())
        oos_mask = ~(X_oos.isna().any(axis=1) | y_oos.isna())
        
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        X_val, y_val = X_val[val_mask], y_val[val_mask]
        X_oos, y_oos = X_oos[oos_mask], y_oos[oos_mask]
        
        print(f"  - Train samples: {len(X_train)}")
        print(f"  - Val samples: {len(X_val)}")
        print(f"  - OOS samples: {len(X_oos)}")
        
        # 訓練多個模型
        n_models = self.config.ensemble_models if self.config.use_ensemble else 1
        self.models = []
        
        for i in range(n_models):
            print(f"  - Training model {i+1}/{n_models}...")
            
            # XGBoost參數
            params = {
                'max_depth': self.config.max_depth,
                'learning_rate': self.config.learning_rate,
                'n_estimators': self.config.n_estimators,
                'min_child_weight': self.config.min_child_weight,
                'subsample': self.config.subsample,
                'colsample_bytree': self.config.colsample_bytree,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': 42 + i,
                'n_jobs': -1,
                'missing': np.nan  # 明確設置missing值
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            self.models.append(model)
        
        # 集成預測
        val_preds = self._ensemble_predict(X_val)
        oos_preds = self._ensemble_predict(X_oos)
        
        # 評估
        val_metrics = self._calculate_metrics(y_val, val_preds)
        oos_metrics = self._calculate_metrics(y_oos, oos_preds)
        
        print(f"\n  [Val Metrics]")
        print(f"    Accuracy: {val_metrics['accuracy']:.3f}")
        print(f"    Precision: {val_metrics['precision']:.3f}")
        print(f"    Recall: {val_metrics['recall']:.3f}")
        print(f"    AUC: {val_metrics['auc']:.3f}")
        
        print(f"\n  [OOS Metrics] *** IMPORTANT ***")
        print(f"    Accuracy: {oos_metrics['accuracy']:.3f}")
        print(f"    Precision: {oos_metrics['precision']:.3f}")
        print(f"    Recall: {oos_metrics['recall']:.3f}")
        print(f"    AUC: {oos_metrics['auc']:.3f}")
        
        # 特徵重要性
        feature_importance = self._get_feature_importance(feature_names)
        
        return {
            'val_metrics': val_metrics,
            'oos_metrics': oos_metrics,
            'feature_importance': feature_importance,
            'n_models': n_models
        }
    
    def _clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        清理數據: 處理inf和極端值
        """
        X = X.copy()
        
        # 替換inf為nan
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # 對於數值列,用中位數填充nan
        for col in X.columns:
            if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                # 填充nan
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                
                # Clip極端值 (±5個標準差)
                if X[col].std() > 0:
                    mean = X[col].mean()
                    std = X[col].std()
                    X[col] = X[col].clip(mean - 5*std, mean + 5*std)
        
        return X
    
    def _ensemble_predict(self, X) -> np.ndarray:
        """
        集成預測 - 投票機制
        """
        if len(self.models) == 1:
            return self.models[0].predict(X)
        
        # 多模型投票
        predictions = np.array([model.predict(X) for model in self.models])
        ensemble_pred = (predictions.mean(axis=0) >= 0.5).astype(int)
        
        return ensemble_pred
    
    def _ensemble_predict_proba(self, X) -> np.ndarray:
        """
        集成概率預測
        """
        if len(self.models) == 1:
            return self.models[0].predict_proba(X)[:, 1]
        
        # 多模型平均概率
        probas = np.array([model.predict_proba(X)[:, 1] for model in self.models])
        ensemble_proba = probas.mean(axis=0)
        
        return ensemble_proba
    
    def _calculate_metrics(self, y_true, y_pred) -> dict:
        """
        計算評估指標
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'auc': float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.5,
            'confusion_matrix': cm.tolist(),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        }
    
    def _get_feature_importance(self, feature_names) -> list:
        """
        獲取特徵重要性
        """
        importance_dict = {}
        
        for model in self.models:
            for feat, imp in zip(feature_names, model.feature_importances_):
                if feat not in importance_dict:
                    importance_dict[feat] = []
                importance_dict[feat].append(imp)
        
        # 平均
        avg_importance = {feat: np.mean(imps) for feat, imps in importance_dict.items()}
        
        # 排序
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_features[:20]
    
    def _save_models(self, results) -> Path:
        """
        保存模型和配置
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path('models') / f"{self.config.symbol}_{self.config.timeframe}_v3_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        for i, model in enumerate(self.models):
            joblib.dump(model, model_dir / f'model_{i}.pkl')
        
        # 保存配置
        joblib.dump(self.config.to_dict(), model_dir / 'config.pkl')
        
        # 保存特徵名
        joblib.dump(self.feature_names, model_dir / 'features.pkl')
        
        # 保存結果
        joblib.dump(results, model_dir / 'results.pkl')
        
        return model_dir
