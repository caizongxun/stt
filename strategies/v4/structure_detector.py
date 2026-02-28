"""
V4 Structure Detector
V4結構識別器 - 支撑/壓力
"""
import pandas as pd
import numpy as np

class StructureDetector:
    """識別支撑壓力和關鍵價位"""
    
    def __init__(self, config):
        self.config = config
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        識別市場結構
        """
        df = df.copy()
        
        window = self.config.support_resistance_window
        tolerance = self.config.key_level_tolerance
        
        # 支撑壓力
        df['support'] = df['low'].rolling(window).min()
        df['resistance'] = df['high'].rolling(window).max()
        
        # 區間寬度
        df['range_width'] = df['resistance'] - df['support']
        df['range_width_pct'] = df['range_width'] / df['close']
        
        # 價格在區間位置 (0=支撑, 1=壓力)
        df['position_in_range'] = (
            (df['close'] - df['support']) / 
            df['range_width'].replace(0, np.nan)
        )
        df['position_in_range'] = df['position_in_range'].fillna(0.5).clip(0, 1)
        
        # 觸碰支撑/壓力
        df['near_support'] = (
            abs(df['close'] - df['support']) / df['close'] < tolerance
        ).astype(int)
        
        df['near_resistance'] = (
            abs(df['close'] - df['resistance']) / df['close'] < tolerance
        ).astype(int)
        
        # 突破/跌破
        df['breakout_up'] = (
            (df['close'] > df['resistance']) &
            (df['close'].shift(1) <= df['resistance'].shift(1))
        ).astype(int)
        
        df['breakout_down'] = (
            (df['close'] < df['support']) &
            (df['close'].shift(1) >= df['support'].shift(1))
        ).astype(int)
        
        return df
