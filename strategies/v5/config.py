"""
V5 Config
"""
from dataclasses import dataclass, asdict

@dataclass
class V5Config:
    # Basic
    symbol: str = 'BTCUSDT'
    timeframe: str = '15m'
    capital: float = 10000
    
    # Label
    forward_bars: int = 8
    min_return_pct: float = 0.008
    require_no_reverse: bool = True
    
    # Features
    lookback_windows: list = None
    use_price_features: bool = True
    use_volume_features: bool = True
    use_volatility_features: bool = True
    use_momentum_features: bool = True
    
    # Model
    max_depth: int = 6
    learning_rate: float = 0.1
    n_estimators: int = 300
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    gamma: float = 0.1
    ensemble_models: int = 5
    
    # Split
    train_size: float = 0.6
    val_size: float = 0.2
    
    # Backtest Thresholds
    long_threshold: float = 0.6
    short_threshold: float = 0.6
    
    # Risk
    leverage: int = 5
    position_pct: float = 0.4
    use_compound: bool = True
    atr_sl_multiplier: float = 2.0
    atr_tp_multiplier: float = 4.0
    use_trailing_stop: bool = True
    trailing_activation: float = 0.015
    trailing_distance: float = 0.008
    max_hold_bars: int = 96
    fee_rate: float = 0.0006
    slippage: float = 0.0002
    
    # Position
    max_positions: int = 3
    max_trades_per_day: int = 10
    min_bars_between: int = 3
    
    def __post_init__(self):
        if self.lookback_windows is None:
            self.lookback_windows = [5, 10, 20, 40]
    
    def to_dict(self):
        d = asdict(self)
        return d
