"""
V4 Signal Generator
V4信號生成器 - 雙模式
"""
import pandas as pd
import numpy as np
import ta

class DualModeSignalGenerator:
    """雙模式信號生成器"""
    
    def __init__(self, config):
        self.config = config
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成信號
        """
        df = df.copy()
        
        # 計算指標
        df = self._calculate_indicators(df)
        
        # 盤整模式信號
        df = self._ranging_signals(df)
        
        # 突破模式信號
        df = self._breakout_signals(df)
        
        # 根據市場狀態選擇信號
        df = self._select_signals(df)
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算技術指標"""
        # RSI
        if 'rsi' not in df.columns:
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # ATR
        if 'atr' not in df.columns:
            df['atr'] = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'], window=self.config.atr_window
            )
        
        # 成交量
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd_diff'] = macd.macd_diff()
        
        return df
    
    def _ranging_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        盤整模式: 區間反轉
        在支撑買,在壓力賣
        """
        # 做多: 支撑附近 + 超賣
        range_long = (
            (df['near_support'] == 1) &
            (df['rsi'] < self.config.range_entry_rsi_low) &
            (df['macd_diff'] > 0)  # MACD轉正
        )
        df['signal_range_long'] = range_long.astype(int)
        
        # 做空: 壓力附近 + 超買
        range_short = (
            (df['near_resistance'] == 1) &
            (df['rsi'] > self.config.range_entry_rsi_high) &
            (df['macd_diff'] < 0)  # MACD轉負
        )
        df['signal_range_short'] = range_short.astype(int)
        
        return df
    
    def _breakout_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        突破模式: 跟隨突破
        突破關鍵價位 + 成交量確認
        """
        # 向上突破
        breakout_long = (
            (df['breakout_up'] == 1) &
            (df['volume_ratio'] > self.config.breakout_volume_multiplier) &
            (df['rsi'] > 50)  # 動量確認
        )
        df['signal_breakout_long'] = breakout_long.astype(int)
        
        # 向下突破
        breakout_short = (
            (df['breakout_down'] == 1) &
            (df['volume_ratio'] > self.config.breakout_volume_multiplier) &
            (df['rsi'] < 50)
        )
        df['signal_breakout_short'] = breakout_short.astype(int)
        
        return df
    
    def _select_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根據市場狀態選擇信號
        """
        # 盤整時用反轉信號
        ranging_mask = df['regime'] == 'RANGING'
        df.loc[ranging_mask, 'signal_long'] = df.loc[ranging_mask, 'signal_range_long']
        df.loc[ranging_mask, 'signal_short'] = df.loc[ranging_mask, 'signal_range_short']
        
        # 趨勢時用突破信號
        trending_mask = df['regime'] == 'TRENDING'
        df.loc[trending_mask, 'signal_long'] = df.loc[trending_mask, 'signal_breakout_long']
        df.loc[trending_mask, 'signal_short'] = df.loc[trending_mask, 'signal_breakout_short']
        
        # 預設0
        if 'signal_long' not in df.columns:
            df['signal_long'] = 0
            df['signal_short'] = 0
        
        df['signal_long'] = df['signal_long'].fillna(0).astype(int)
        df['signal_short'] = df['signal_short'].fillna(0).astype(int)
        
        return df
