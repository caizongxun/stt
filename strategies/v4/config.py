"""
V4 Configuration - Adaptive Dual Mode
V4配置 - 自適應雙模式
"""
from dataclasses import dataclass, field

@dataclass
class V4Config:
    """
    V4策略: 盤整反轉 + 突破跟隨
    根據市場狀態自動切換
    """
    # 數據配置
    symbol: str = "BTCUSDT"
    timeframe: str = "15m"
    
    # OOS驗證
    train_size: float = 0.75
    val_size: float = 0.125
    oos_size: float = 0.125
    
    # 市場狀態分類
    adx_ranging_threshold: float = 25.0  # ADX<25 = 盤整
    adx_trending_threshold: float = 25.0  # ADX>25 = 趨勢
    bb_squeeze_percentile: float = 0.25  # BB寬度<25%分位 = 低波
    
    # 支撑壓力識別
    support_resistance_window: int = 20  # 識別窗口
    key_level_tolerance: float = 0.02  # 2%內算觸碰
    
    # 盤整模式參數
    range_entry_rsi_low: float = 30
    range_entry_rsi_high: float = 70
    range_target_pct: float = 0.5  # 目標區間的50%
    
    # 突破模式參數
    breakout_volume_multiplier: float = 1.5  # 成交量1.5倍
    breakout_confirmation_bars: int = 2  # 確認2根K棒
    
    # ATR風控
    atr_window: int = 14
    atr_sl_multiplier: float = 1.5
    atr_tp_range: float = 1.5  # 盤整模式
    atr_tp_breakout: float = 3.0  # 突破模式
    
    # 模型參數
    model_type: str = "xgboost"
    max_depth: int = 5
    learning_rate: float = 0.05
    n_estimators: int = 200
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    # 集成學習
    use_ensemble: bool = True
    ensemble_models: int = 3
    
    # 預測閉值
    predict_threshold: float = 0.60
    
    # 資金管理
    capital: float = 10000
    leverage: int = 3
    position_pct: float = 0.3
    use_compound: bool = True
    
    # 交易管理
    max_positions: int = 2
    max_trades_per_day: int = 8
    
    # 費用
    fee_rate: float = 0.0006
    slippage: float = 0.0005
    
    # 信號模式 (新增)
    signal_mode: str = 'pure'  # pure/hybrid/ranging/trending
    use_regime_filter: bool = False  # 是否用狀態過濾
    use_rsi_filter: bool = False  # 是否用RSI過濾
    use_volume_filter: bool = False  # 是否用成交量過濾
    use_support_resistance_filter: bool = False  # 是否用支撑/壓力過濾
    
    def to_dict(self) -> dict:
        return self.__dict__.copy()
