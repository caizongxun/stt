"""
V3 Configuration - Aggressive High Performance
V3配置 - 激進高性能系統
"""
from dataclasses import dataclass

@dataclass
class V3Config:
    """
    V3策略: 目標30天50%報酬
    核心: 高頻+槓桶+複利+多策略
    """
    # 數據配置
    symbol: str = "BTCUSDT"
    timeframe: str = "15m"
    
    # OOS驗證參數
    train_months: int = 6  # 訓練6個月
    val_months: int = 1    # 驗證1個月
    oos_months: int = 1    # OOS測試1個月
    
    # 策略組合
    use_bb_reversal: bool = True      # BB反轉
    use_momentum_breakout: bool = True # 動量突破
    use_trend_following: bool = True   # 趨勢跟隨
    
    # BB反轉參數
    bb_window: int = 20
    bb_std: float = 2.0
    bb_touch_threshold: float = 0.02  # 2%內算觸碰
    bb_reversal_atr: float = 1.5      # 放寬標準(提高信號)
    
    # 動量突破參數
    momentum_window: int = 10
    momentum_threshold: float = 0.015  # 1.5%突破
    volume_surge: float = 2.0          # 成交量2倍
    
    # 趨勢跟隨參數
    trend_fast_ma: int = 10
    trend_slow_ma: int = 30
    trend_min_strength: float = 0.02   # 2%趨勢強度
    
    # ATR風控
    atr_window: int = 14
    atr_sl_multiplier: float = 1.5     # 緊止損
    atr_tp_multiplier: float = 2.5     # 積極止盈
    trailing_stop_atr: float = 1.0     # 移動止損
    
    # 模型參數 - 提高複雜度
    model_type: str = "xgboost"  # XGBoost比LightGBM更強
    max_depth: int = 6
    learning_rate: float = 0.05
    n_estimators: int = 300
    min_child_weight: int = 3
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    # 集成學習
    use_ensemble: bool = True
    ensemble_models: int = 5  # 5個模型投票
    
    # 預測閉值 - 動態調整
    predict_threshold_base: float = 0.60
    use_dynamic_threshold: bool = True  # 根據市場波動調整
    
    # 資金管理 - 激進
    capital: float = 10000
    leverage: int = 5              # 5倍槓桶
    use_compound: bool = True      # 複利模式
    position_pct: float = 0.30     # 30%仓位(含槓桶)
    max_risk_per_trade: float = 0.05  # 5%單筆風險
    
    # 交易管理
    max_positions: int = 3         # 最多3個同時仓位
    max_trades_per_day: int = 10   # 每姑10筆
    min_bars_between: int = 2      # 最少間4筆交易
    
    # 時間管理
    avoid_weekend: bool = True     # 避開周末
    trading_hours: list = None     # 交易時段範圍
    
    # 止損機制
    use_time_stop: bool = True     # 時間止損
    max_hold_bars: int = 48        # 12小時強制平倉
    daily_loss_limit: float = 0.10 # 單日10%止損
    total_drawdown_limit: float = 0.20  # 20%總回撤止損
    
    # 費用
    fee_rate: float = 0.0006       # 0.06% (幣安合約)
    slippage: float = 0.0005       # 0.05%滑點
    
    def __post_init__(self):
        if self.trading_hours is None:
            # 亞洲+歐洲時段 (UTC)
            self.trading_hours = [(0, 10), (13, 23)]  # 避開美國休市
    
    def to_dict(self) -> dict:
        return self.__dict__.copy()
