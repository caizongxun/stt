"""
V4 Label Generator
V4標籤生成器 - 自適應標籤
"""
import pandas as pd
import numpy as np

class AdaptiveLabelGenerator:
    """根據市場狀態生成不同標籤"""
    
    def __init__(self, config):
        self.config = config
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成自適應標籤
        """
        df = df.copy()
        
        # 計算未來價格
        df = self._calculate_future_prices(df)
        
        # 盤整模式標籤
        df = self._ranging_labels(df)
        
        # 突破模式標籤
        df = self._breakout_labels(df)
        
        # 合併標籤
        df = self._merge_labels(df)
        
        return df
    
    def _calculate_future_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算未來價格"""
        # 未來10根K棒的高低點
        df['future_high_10'] = df['high'].shift(-10).rolling(10).max()
        df['future_low_10'] = df['low'].shift(-10).rolling(10).min()
        
        return df
    
    def _ranging_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        盤整模式標籤
        目標: 到達區間另一端
        """
        # 做多標籤: 從支撑到中間/壓力
        range_target_long = df['support'] + df['range_width'] * self.config.range_target_pct
        
        df['label_range_long'] = (
            (df['future_high_10'] >= range_target_long) &  # 達到目標
            (df['future_low_10'] >= df['close'] - df['atr'] * self.config.atr_sl_multiplier)  # 沒有大幅回撤
        ).astype(int)
        
        # 做空標籤: 從壓力到中間/支撑
        range_target_short = df['resistance'] - df['range_width'] * self.config.range_target_pct
        
        df['label_range_short'] = (
            (df['future_low_10'] <= range_target_short) &
            (df['future_high_10'] <= df['close'] + df['atr'] * self.config.atr_sl_multiplier)
        ).astype(int)
        
        return df
    
    def _breakout_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        突破模式標籤
        目標: 趨勢延續
        """
        # 做多標籤: 向上突破後繼續上漲
        breakout_target_long = df['close'] + df['atr'] * self.config.atr_tp_breakout
        
        df['label_breakout_long'] = (
            (df['future_high_10'] >= breakout_target_long) &
            (df['future_low_10'] >= df['close'] - df['atr'] * self.config.atr_sl_multiplier)
        ).astype(int)
        
        # 做空標籤: 向下突破後繼續下跌
        breakout_target_short = df['close'] - df['atr'] * self.config.atr_tp_breakout
        
        df['label_breakout_short'] = (
            (df['future_low_10'] <= breakout_target_short) &
            (df['future_high_10'] <= df['close'] + df['atr'] * self.config.atr_sl_multiplier)
        ).astype(int)
        
        return df
    
    def _merge_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根據市場狀態合併標籤
        """
        # 盤整時用反轉標籤
        df['label_long'] = 0
        df['label_short'] = 0
        
        ranging_mask = df['regime'] == 'RANGING'
        df.loc[ranging_mask, 'label_long'] = df.loc[ranging_mask, 'label_range_long']
        df.loc[ranging_mask, 'label_short'] = df.loc[ranging_mask, 'label_range_short']
        
        # 趨勢時用突破標籤
        trending_mask = df['regime'] == 'TRENDING'
        df.loc[trending_mask, 'label_long'] = df.loc[trending_mask, 'label_breakout_long']
        df.loc[trending_mask, 'label_short'] = df.loc[trending_mask, 'label_breakout_short']
        
        # 總標籤
        df['label'] = ((df['label_long'] == 1) | (df['label_short'] == 1)).astype(int)
        
        return df
    
    def get_statistics(self, df: pd.DataFrame) -> dict:
        """獲取標籤統計"""
        total = len(df.dropna(subset=['label']))
        positive = df['label'].sum()
        long_labels = df['label_long'].sum()
        short_labels = df['label_short'].sum()
        
        # 分狀態統計
        ranging = df[df['regime'] == 'RANGING']
        trending = df[df['regime'] == 'TRENDING']
        
        return {
            'total_samples': total,
            'positive_labels': int(positive),
            'positive_rate': float(positive / total * 100) if total > 0 else 0,
            'long_labels': int(long_labels),
            'short_labels': int(short_labels),
            'ranging_positive': int(ranging['label'].sum()) if len(ranging) > 0 else 0,
            'ranging_rate': float(ranging['label'].sum() / len(ranging) * 100) if len(ranging) > 0 else 0,
            'trending_positive': int(trending['label'].sum()) if len(trending) > 0 else 0,
            'trending_rate': float(trending['label'].sum() / len(trending) * 100) if len(trending) > 0 else 0
        }
