"""
V5 Configuration - Pure ML Price Prediction
V5配置 - 純ML價格預測

目標: 月報酬 20-50%
理念: 讓模型自由預測,不用任何硬規則
"""
from dataclasses import dataclass

@dataclass
class V5Config:
    """
    V5: 極簡高效系統
    - 只預測價格漲/跌
    - 不用市場狀態
    - 不用支撑/壓力
    - 完全信任模型
    """
    
    # 基本配置
    symbol: str = "BTCUSDT"
    timeframe: str = "15m"
    
    # 標籤參數
    forward_bars: int = 8  # 預測未來8根K棒(2小時)
    min_return_pct: float = 0.008  # 最小目標0.8%(含手續費)
    require_no_reverse: bool = True  # 要求中間不回撤
    
    # 特徵配置
    use_price_features: bool = True  # 價格特徵
    use_volume_features: bool = True  # 成交量特徵
    use_volatility_features: bool = True  # 波動率特徵
    use_momentum_features: bool = True  # 動量特徵
    
    lookback_windows: list = None  # [5, 10, 20, 40]
    
    def __post_init__(self):
        if self.lookback_windows is None:
            self.lookback_windows = [5, 10, 20, 40]
    
    # 數據分割
    train_size: float = 0.70
    val_size: float = 0.15
    oos_size: float = 0.15
    
    # 模型參數
    model_type: str = "xgboost"
    max_depth: int = 6  # 提高到6
    learning_rate: float = 0.03  # 降低學習率
    n_estimators: int = 300  # 增加樹數量
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    gamma: float = 0.1  # 正則化
    
    # 集成學習
    use_ensemble: bool = True
    ensemble_models: int = 5  # 5個模型投票
    
    # 預測閉值
    predict_threshold: float = 0.55  # 預設0.55
    
    # 資金管理
    capital: float = 10000
    leverage: int = 5  # 5倍槓桶(中等風險)
    position_pct: float = 0.4  # 40%仓位
    use_compound: bool = True  # 複利
    
    # 風控參數
    atr_sl_multiplier: float = 2.0  # 2 ATR止損
    atr_tp_multiplier: float = 4.0  # 4 ATR止盈
    use_trailing_stop: bool = True  # 移動止損
    trailing_activation: float = 0.015  # 1.5%啟動
    trailing_distance: float = 0.008  # 0.8%距離
    
    # 交易限制
    max_positions: int = 3  # 最多3個持仓
    max_trades_per_day: int = 12  # 每天最多12筆
    min_bars_between: int = 2  # 間隔2根K棒
    
    # 時間止損
    max_hold_bars: int = 32  # 最长持有32根K棒(8小時)
    
    # 費用
    fee_rate: float = 0.0006  # 0.06%
    slippage: float = 0.0005  # 0.05%
    
    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                result[key] = value
        return result
