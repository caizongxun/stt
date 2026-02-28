"""
V1 Configuration
V1配置檔
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class V1Config:
    """
    V1策略配置
    """
    # 數據配置
    symbol: str = "BTCUSDT"
    timeframe: str = "15m"
    train_size: float = 0.7
    
    # LightGBM參數 - 減少過擬合
    num_leaves: int = 20  # 31 -> 20
    max_depth: int = 4    # 6 -> 4
    learning_rate: float = 0.05
    n_estimators: int = 150  # 100 -> 150
    min_child_samples: int = 50  # 新增正則化
    
    # 類別不平衡處理 - 降低權重
    use_class_weight: bool = True
    class_weights: dict = None
    
    # 標籤生成 - 提高閘值提升精確度
    label_threshold_long: float = 0.008   # 0.005 -> 0.008
    label_threshold_short: float = -0.008 # -0.005 -> -0.008
    label_periods: int = 3
    
    # 特徵工程 - 增加更多特徵
    lookback_periods: list = None
    use_volume_features: bool = True  # 新增成交量特徵
    
    # 回測參數
    capital: float = 10000
    leverage: int = 1
    fee_rate: float = 0.001
    backtest_days: int = 90
    
    def __post_init__(self):
        if self.lookback_periods is None:
            self.lookback_periods = [5, 10, 20, 50]  # 增加50期
        
        if self.class_weights is None and self.use_class_weight:
            # 優化: 降低權重避免過度預測
            self.class_weights = {
                0: 1.0,   # hold
                1: 5.0,   # long (15 -> 5)
                2: 5.0    # short (15 -> 5)
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
