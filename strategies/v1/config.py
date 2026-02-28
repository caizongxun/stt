"""
V1 Configuration - Emergency Fix
V1配置檔 - 緊急修復
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class V1Config:
    """
    V1策略配置 - 緊急修復版
    """
    # 數據配置
    symbol: str = "BTCUSDT"
    timeframe: str = "15m"
    train_size: float = 0.7
    
    # LightGBM參數 - 大幅增加正則化
    num_leaves: int = 15       # 20 -> 15
    max_depth: int = 3         # 4 -> 3
    learning_rate: float = 0.03  # 0.05 -> 0.03
    n_estimators: int = 100     # 150 -> 100
    min_child_samples: int = 100  # 50 -> 100
    
    # 類別權重 - 降低交易頻率
    use_class_weight: bool = True
    class_weights: dict = None
    
    # 標籤閘值 - 大幅提高減少交易
    label_threshold_long: float = 0.015   # 0.6% -> 1.5%
    label_threshold_short: float = -0.015 # -0.6% -> -1.5%
    label_periods: int = 5     # 3 -> 5
    
    # 特徵工程
    lookback_periods: list = None
    use_volume_features: bool = True
    
    # 回測參數
    capital: float = 10000
    leverage: int = 1
    fee_rate: float = 0.001
    backtest_days: int = 90
    
    def __post_init__(self):
        if self.lookback_periods is None:
            self.lookback_periods = [10, 20, 50, 100]  # 使用更長周期
        
        if self.class_weights is None and self.use_class_weight:
            # 降低權重減少過度預測
            self.class_weights = {
                0: 1.0,   # hold
                1: 3.0,   # long (8 -> 3)
                2: 3.0    # short (8 -> 3)
            }
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'train_size': self.train_size,
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'min_child_samples': self.min_child_samples,
            'use_class_weight': self.use_class_weight,
            'class_weights': self.class_weights,
            'label_threshold_long': self.label_threshold_long,
            'label_threshold_short': self.label_threshold_short,
            'label_periods': self.label_periods,
            'lookback_periods': self.lookback_periods,
            'use_volume_features': self.use_volume_features,
            'capital': self.capital,
            'leverage': self.leverage,
            'fee_rate': self.fee_rate,
            'backtest_days': self.backtest_days
        }
