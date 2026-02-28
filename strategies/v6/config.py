"""
V6 Config - Reversal Prediction Model
"""
from dataclasses import dataclass, asdict


@dataclass
class V6Config:
    # Data
    symbol: str = "BTCUSDT"
    timeframe: str = "15m"
    train_size: float = 0.7
    val_size: float = 0.15
    forward_bars: int = 6
    reversal_target_pct: float = 0.006
    oversold_rsi: int = 32
    overbought_rsi: int = 68
    bb_std: float = 2.1
    min_bb_position: float = 0.25

    # Model
    num_leaves: int = 48
    learning_rate: float = 0.05
    n_estimators: int = 180
    max_depth: int = 6
    subsample: float = 0.9
    colsample_bytree: float = 0.8

    # Thresholds
    long_threshold: float = 0.55
    short_threshold: float = 0.55

    # Risk
    capital: float = 10000
    leverage: int = 2
    position_pct: float = 0.25
    fee_rate: float = 0.0004
    slippage: float = 0.00015
    atr_sl_multiplier: float = 1.6
    atr_tp_multiplier: float = 2.8
    max_hold_bars: int = 48
    use_compound: bool = True

    def to_dict(self):
        return asdict(self)
