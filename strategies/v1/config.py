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
    
    # LightGBM參數
    num_leaves: int = 31
    max_depth: int = 6
    learning_rate: float = 0.05
    n_estimators: int = 100
    
    # 特徵工程
    lookback_periods: list = None
    
    # 回測參數
    capital: float = 10000
    leverage: int = 1
    fee_rate: float = 0.001
    backtest_days: int = 90
    
    def __post_init__(self):
        if self.lookback_periods is None:
            self.lookback_periods = [5, 10, 20]
    
    def to_dict(self) -> dict:
        """轉換為字典"""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'train_size': self.train_size,
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'lookback_periods': self.lookback_periods,
            'capital': self.capital,
            'leverage': self.leverage,
            'fee_rate': self.fee_rate,
            'backtest_days': self.backtest_days
        }
