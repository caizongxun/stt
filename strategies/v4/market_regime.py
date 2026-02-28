"""
V4 Market Regime Detector
V4市場狀態識別器
"""
import pandas as pd
import numpy as np
import ta

class MarketRegimeDetector:
    """市場狀態分類: 盤整 vs 趨勢"""
    
    def __init__(self, config):
        self.config = config
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        識別市場狀態
        返回: df with regime columns
        """
        df = df.copy()
        
        # 計算ADX
        if 'adx' not in df.columns:
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        
        # 計算BB寬度
        if 'bb_width' not in df.columns:
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # BB寬度百分位
        df['bb_width_percentile'] = df['bb_width'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )
        
        # 市場狀態分類
        df['regime'] = 'UNKNOWN'
        
        # 盤整: ADX低 + BB窄
        ranging = (
            (df['adx'] < self.config.adx_ranging_threshold) &
            (df['bb_width_percentile'] < self.config.bb_squeeze_percentile)
        )
        df.loc[ranging, 'regime'] = 'RANGING'
        
        # 趨勢: ADX高
        trending = (df['adx'] >= self.config.adx_trending_threshold)
        df.loc[trending, 'regime'] = 'TRENDING'
        
        # 狀態編碼
        df['regime_code'] = 0
        df.loc[df['regime'] == 'RANGING', 'regime_code'] = 1
        df.loc[df['regime'] == 'TRENDING', 'regime_code'] = 2
        
        return df
    
    def get_regime_statistics(self, df: pd.DataFrame) -> dict:
        """獲取狀態統計"""
        total = len(df)
        ranging = (df['regime'] == 'RANGING').sum()
        trending = (df['regime'] == 'TRENDING').sum()
        unknown = (df['regime'] == 'UNKNOWN').sum()
        
        return {
            'total': total,
            'ranging': int(ranging),
            'ranging_pct': float(ranging / total * 100) if total > 0 else 0,
            'trending': int(trending),
            'trending_pct': float(trending / total * 100) if total > 0 else 0,
            'unknown': int(unknown),
            'unknown_pct': float(unknown / total * 100) if total > 0 else 0
        }
