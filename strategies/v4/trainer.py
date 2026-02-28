"""
V4 Trainer
V4訓練器
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import joblib
from pathlib import Path
from datetime import datetime

from .market_regime import MarketRegimeDetector
from .structure_detector import StructureDetector
from .signal_generator import DualModeSignalGenerator
from .label_generator import AdaptiveLabelGenerator

class V4Trainer:
    """
V4訓練器
    """
    
    def __init__(self, config):
        self.config = config
        self.models = []
        self.feature_names = []
    
    def train(self, df: pd.DataFrame) -> dict:
        """完整訓練流程"""
        print("\n[V4 Adaptive Training]")
        
        # 1. 市場狀態識別
        print("Step 1: Market regime detection...")
        regime_detector = MarketRegimeDetector(self.config)
        df = regime_detector.detect(df)
        regime_stats = regime_detector.get_regime_statistics(df)
        print(f"  - Ranging: {regime_stats['ranging_pct']:.1f}%")
        print(f"  - Trending: {regime_stats['trending_pct']:.1f}%")
        
        # 2. 結構識別
        print("Step 2: Structure detection...")
        structure_detector = StructureDetector(self.config)
        df = structure_detector.detect(df)
        
        # 3. 信號生成
        print("Step 3: Signal generation...")
        signal_gen = DualModeSignalGenerator(self.config)
        df = signal_gen.generate(df)
        
        # 4. 標籤生成
        print("Step 4: Adaptive label generation...")
        label_gen = AdaptiveLabelGenerator(self.config)
        df = label_gen.generate(df)
        label_stats = label_gen.get_statistics(df)
        print(f"  - Positive rate: {label_stats['positive_rate']:.1f}%")
        print(f"  - Ranging positive: {label_stats['ranging_positive']} ({label_stats['ranging_rate']:.1f}%)")
        print(f"  - Trending positive: {label_stats['trending_positive']} ({label_stats['trending_rate']:.1f}%)")
        
        # 5. 特徵工程
        print("Step 5: Feature engineering...")
        df, feature_names = self._engineer_features(df)
        self.feature_names = feature_names
        print(f"  - Features: {len(feature_names)}")
        
        # 6. 數據分割
        print("Step 6: Data split...")
        train_df, val_df, oos_df = self._split_data(df)
        print(f"  - Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  - Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  - OOS: {len(oos_df)} ({len(oos_df)/len(df)*100:.1f}%)")
        
        # 7. 訓練模型
        print("Step 7: Training models...")
        results = self._train_models(train_df, val_df, oos_df, feature_names)
        
        # 8. 保存
        print("Step 8: Saving...")
        model_path = self._save_models(results)
        
        # 返回結果
        results['regime_statistics'] = regime_stats
        results['label_statistics'] = label_stats
        results['model_path'] = str(model_path)
        
        return results
    
    def _engineer_features(self, df: pd.DataFrame) -> tuple:
        """特徵工程"""
        features = [
            # 市場狀態
            'regime_code', 'adx', 'bb_width', 'bb_width_percentile',
            
            # 結構
            'position_in_range', 'range_width_pct',
            'near_support', 'near_resistance',
            
            # 技術指標
            'rsi', 'macd_diff', 'atr',
            
            # 成交量
            'volume_ratio',
            
            # 信號
            'signal_long', 'signal_short'
        ]
        
        # 確保所有特徵存在
        available_features = [f for f in features if f in df.columns]
        
        return df, available_features
    
    def _split_data(self, df: pd.DataFrame) -> tuple:
        """數據分割"""
        total = len(df)
        train_end = int(total * self.config.train_size)
        val_end = int(total * (self.config.train_size + self.config.val_size))
        
        return (
            df.iloc[:train_end].copy(),
            df.iloc[train_end:val_end].copy(),
            df.iloc[val_end:].copy()
        )
    
    def _train_models(self, train_df, val_df, oos_df, feature_names) -> dict:
        """訓練模型"""
        # 準備數據
        X_train, y_train = self._prepare_data(train_df, feature_names)
        X_val, y_val = self._prepare_data(val_df, feature_names)
        X_oos, y_oos = self._prepare_data(oos_df, feature_names)
        
        print(f"  - Train: {len(X_train)} samples")
        print(f"  - Val: {len(X_val)} samples")
        print(f"  - OOS: {len(X_oos)} samples")
        
        # 訓練模型
        n_models = self.config.ensemble_models if self.config.use_ensemble else 1
        
        for i in range(n_models):
            print(f"  - Training model {i+1}/{n_models}...")
            
            model = xgb.XGBClassifier(
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                n_estimators=self.config.n_estimators,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                objective='binary:logistic',
                eval_metric='auc',
                random_state=42 + i,
                missing=np.nan
            )
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            self.models.append(model)
        
        # 評估
        val_preds = self._ensemble_predict(X_val)
        oos_preds = self._ensemble_predict(X_oos)
        
        val_metrics = self._calculate_metrics(y_val, val_preds)
        oos_metrics = self._calculate_metrics(y_oos, oos_preds)
        
        print(f"\n  [Val] Acc:{val_metrics['accuracy']:.3f} Prec:{val_metrics['precision']:.3f} Rec:{val_metrics['recall']:.3f} AUC:{val_metrics['auc']:.3f}")
        print(f"  [OOS] Acc:{oos_metrics['accuracy']:.3f} Prec:{oos_metrics['precision']:.3f} Rec:{oos_metrics['recall']:.3f} AUC:{oos_metrics['auc']:.3f}")
        
        # 特徵重要性
        feature_importance = self._get_feature_importance(feature_names)
        
        return {
            'val_metrics': val_metrics,
            'oos_metrics': oos_metrics,
            'feature_importance': feature_importance
        }
    
    def _prepare_data(self, df: pd.DataFrame, feature_names: list) -> tuple:
        """準備數據"""
        X = df[feature_names].copy()
        y = df['label'].copy()
        
        # 清理
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())
            if X[col].std() > 0:
                mean, std = X[col].mean(), X[col].std()
                X[col] = X[col].clip(mean - 5*std, mean + 5*std)
        
        # 移除NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        return X[mask], y[mask]
    
    def _ensemble_predict(self, X) -> np.ndarray:
        """集成預測"""
        if len(self.models) == 1:
            return self.models[0].predict(X)
        predictions = np.array([m.predict(X) for m in self.models])
        return (predictions.mean(axis=0) >= 0.5).astype(int)
    
    def _calculate_metrics(self, y_true, y_pred) -> dict:
        """計算指標"""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'auc': float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.5,
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        }
    
    def _get_feature_importance(self, feature_names) -> list:
        """特徵重要性"""
        importance = {}
        for model in self.models:
            for feat, imp in zip(feature_names, model.feature_importances_):
                if feat not in importance:
                    importance[feat] = []
                importance[feat].append(imp)
        
        avg_importance = {f: np.mean(imps) for f, imps in importance.items()}
        return sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def _save_models(self, results) -> Path:
        """保存模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path('models') / f"{self.config.symbol}_{self.config.timeframe}_v4_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        for i, model in enumerate(self.models):
            joblib.dump(model, model_dir / f'model_{i}.pkl')
        
        joblib.dump(self.config.to_dict(), model_dir / 'config.pkl')
        joblib.dump(self.feature_names, model_dir / 'features.pkl')
        joblib.dump(results, model_dir / 'results.pkl')
        
        return model_dir
