"""
V2 Configuration
V2配置檔
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class V2Config:
    """
    V2策略配置 - BB反轉系統
    """
    # 數據配置
    symbol: str = "BTCUSDT"
    timeframe: str = "15m"
    train_size: float = 0.7
    
    # BB參數
    bb_window: int = 20
    bb_std: float = 2.0
    
    # ATR參數
    atr_window: int = 14
    atr_sl_multiplier: float = 2.0
    atr_tp_multiplier: float = 3.0
    
    # 標籤生成參數
    reversal_lookforward: int = 10
    min_reversal_atr: float = 1.5
    breakout_tolerance: float = 0.01
    
    # 模型參數 - 優化
    model_type: str = "lightgbm"
    num_leaves: int = 15  # 降低從20->15 (減少過擬合)
    max_depth: int = 3  # 降低從4->3
    learning_rate: float = 0.01  # 降低從0.03->0.01 (更保守)
    n_estimators: int = 200  # 提高從150->200
    min_child_samples: int = 100  # 提高從50->100 (更严格)
    
    # 類別權重 - 重新調整
    use_class_weight: bool = True
    class_weights: dict = None
    
    # 預測阈值 - 提高精準度
    predict_threshold: float = 0.6  # 預測機率>0.6才算有效反轉
    
    # 特徵工程
    use_technical_indicators: bool = True
    use_market_regime: bool = True
    use_historical_success: bool = True
    
    # 回測參數
    capital: float = 10000
    leverage: int = 1
    fee_rate: float = 0.001
    backtest_days: int = 90
    
    # 進場策略
    entry_strategy: str = "hybrid"
    initial_position_pct: float = 0.15
    add_on_position_pct: float = 0.10
    max_add_ons: int = 2
    
    # 風控
    max_risk_per_trade: float = 0.02
    max_concurrent_positions: int = 2
    max_trades_per_day: int = 5
    max_drawdown_stop: float = 0.20
    
    def __post_init__(self):
        if self.class_weights is None and self.use_class_weight:
            # 降低權重從5.0->2.0
            self.class_weights = {
                0: 1.0,
                1: 2.0  # 有效反轉權重降低
            }
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'train_size': self.train_size,
            'bb_window': self.bb_window,
            'bb_std': self.bb_std,
            'atr_window': self.atr_window,
            'atr_sl_multiplier': self.atr_sl_multiplier,
            'atr_tp_multiplier': self.atr_tp_multiplier,
            'reversal_lookforward': self.reversal_lookforward,
            'min_reversal_atr': self.min_reversal_atr,
            'breakout_tolerance': self.breakout_tolerance,
            'model_type': self.model_type,
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'min_child_samples': self.min_child_samples,
            'use_class_weight': self.use_class_weight,
            'class_weights': self.class_weights,
            'predict_threshold': self.predict_threshold,
            'use_technical_indicators': self.use_technical_indicators,
            'use_market_regime': self.use_market_regime,
            'use_historical_success': self.use_historical_success,
            'capital': self.capital,
            'leverage': self.leverage,
            'fee_rate': self.fee_rate,
            'backtest_days': self.backtest_days,
            'entry_strategy': self.entry_strategy,
            'initial_position_pct': self.initial_position_pct,
            'add_on_position_pct': self.add_on_position_pct,
            'max_add_ons': self.max_add_ons,
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_concurrent_positions': self.max_concurrent_positions,
            'max_trades_per_day': self.max_trades_per_day,
            'max_drawdown_stop': self.max_drawdown_stop
        }
