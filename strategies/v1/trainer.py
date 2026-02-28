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
            dict: 訓練結果
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
            verbosity=-1
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='multi_logloss'
        )
        
        # 5. 評估
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)
        
        # 6. 保存
        model_dir = self._save_model()
        
        return {
            'train_score': float(train_score),
            'val_score': float(val_score),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'features': len(self.feature_names),
            'model_path': str(model_dir)
        }
    
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
        
        # 簡單標籤: 未來報酬
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
