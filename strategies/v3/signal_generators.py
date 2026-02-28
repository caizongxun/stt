"""
V3 Signal Generators
V3信號生成器 - 多策略融合
"""
import pandas as pd
import numpy as np
import ta

class SignalGenerator:
    """多策略信號生成器"""
    
    def __init__(self, config):
        self.config = config
    
    def generate_all_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成所有策略信號
        返回: df with signal columns
        """
        df = df.copy()
        
        # 計算基礎指標
        df = self._calculate_indicators(df)
        
        # 策略1: BB反轉
        if self.config.use_bb_reversal:
            df = self._bb_reversal_signals(df)
        
        # 策略2: 動量突破
        if self.config.use_momentum_breakout:
            df = self._momentum_breakout_signals(df)
        
        # 策略3: 趨勢跟隨
        if self.config.use_trend_following:
            df = self._trend_following_signals(df)
        
        # 融合信號
        df = self._combine_signals(df)
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算所有需要的技術指標
        """
        # BB
        bb = ta.volatility.BollingerBands(
            df['close'], 
            window=self.config.bb_window,
            window_dev=self.config.bb_std
        )
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        df['atr'] = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'],
            window=self.config.atr_window
        )
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # ADX
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        
        # 移動平均
        df['ma_fast'] = df['close'].rolling(self.config.trend_fast_ma).mean()
        df['ma_slow'] = df['close'].rolling(self.config.trend_slow_ma).mean()
        
        # 成交量
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # 動量
        df['momentum'] = df['close'].pct_change(self.config.momentum_window)
        df['momentum_strength'] = abs(df['momentum'])
        
        # 波動率
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        
        return df
    
    def _bb_reversal_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        BB反轉信號
        當價格觸碰通道邊界且出現反轉跡象
        """
        threshold = self.config.bb_touch_threshold
        
        # 觸碰上軌 (做空信號)
        touch_upper = df['close'] >= df['bb_upper'] * (1 - threshold)
        reversal_down = (
            (df['rsi'] > 70) &  # 超買
            (df['macd_diff'] < 0) &  # MACD轉弱
            (df['close'] < df['close'].shift(1))  # 開始下跌
        )
        df['signal_bb_short'] = (touch_upper & reversal_down).astype(int)
        
        # 觸碰下軌 (做多信號)
        touch_lower = df['close'] <= df['bb_lower'] * (1 + threshold)
        reversal_up = (
            (df['rsi'] < 30) &  # 超賣
            (df['macd_diff'] > 0) &  # MACD轉強
            (df['close'] > df['close'].shift(1))  # 開始上漲
        )
        df['signal_bb_long'] = (touch_lower & reversal_up).astype(int)
        
        return df
    
    def _momentum_breakout_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        動量突破信號
        突破關鍵價位且伴隨成交量放大
        """
        threshold = self.config.momentum_threshold
        volume_surge = self.config.volume_surge
        
        # 向上突破
        breakout_up = (
            (df['momentum'] > threshold) &  # 強勁上漲
            (df['volume_ratio'] > volume_surge) &  # 成交量放大
            (df['adx'] > 25) &  # 趨勢強度足夠
            (df['close'] > df['bb_middle'])  # 突破中軌
        )
        df['signal_momentum_long'] = breakout_up.astype(int)
        
        # 向下突破
        breakout_down = (
            (df['momentum'] < -threshold) &  # 強勁下跌
            (df['volume_ratio'] > volume_surge) &  # 成交量放大
            (df['adx'] > 25) &  # 趨勢強度足夠
            (df['close'] < df['bb_middle'])  # 跌破中軌
        )
        df['signal_momentum_short'] = breakout_down.astype(int)
        
        return df
    
    def _trend_following_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        趨勢跟隨信號
        順勢交易,捕捉中期趨勢
        """
        min_strength = self.config.trend_min_strength
        
        # 上升趨勢
        uptrend = (
            (df['ma_fast'] > df['ma_slow']) &  # 金叉
            (df['close'] > df['ma_fast']) &  # 價格在快線上方
            (df['adx'] > 20) &  # 趨勢存在
            (df['close'].pct_change(5) > min_strength)  # 趨勢強度
        )
        # 回踩做多
        pullback_long = (
            uptrend &
            (df['rsi'] < 60) &  # 尚未超買
            (df['close'] > df['bb_lower'])  # 不在超賣區
        )
        df['signal_trend_long'] = pullback_long.astype(int)
        
        # 下降趨勢
        downtrend = (
            (df['ma_fast'] < df['ma_slow']) &  # 死叉
            (df['close'] < df['ma_fast']) &  # 價格在快線下方
            (df['adx'] > 20) &  # 趨勢存在
            (df['close'].pct_change(5) < -min_strength)  # 趨勢強度
        )
        # 反彈做空
        pullback_short = (
            downtrend &
            (df['rsi'] > 40) &  # 尚未超賣
            (df['close'] < df['bb_upper'])  # 不在超買區
        )
        df['signal_trend_short'] = pullback_short.astype(int)
        
        return df
    
    def _combine_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        信號融合邏輯
        多個策略同時發出信號時加權
        """
        # 做多信號強度
        df['signal_long_strength'] = (
            df.get('signal_bb_long', 0) * 1.0 +
            df.get('signal_momentum_long', 0) * 1.2 +  # 動量權重更高
            df.get('signal_trend_long', 0) * 0.8
        )
        
        # 做空信號強度
        df['signal_short_strength'] = (
            df.get('signal_bb_short', 0) * 1.0 +
            df.get('signal_momentum_short', 0) * 1.2 +
            df.get('signal_trend_short', 0) * 0.8
        )
        
        # 最終信號 (強度>1視為有效)
        df['has_long_signal'] = (df['signal_long_strength'] >= 1.0).astype(int)
        df['has_short_signal'] = (df['signal_short_strength'] >= 1.0).astype(int)
        
        return df
