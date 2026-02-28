"""
V3 Label Generator
V3標籤生成器 - 未來收益預測
"""
import pandas as pd
import numpy as np

class LabelGenerator:
    """標籤生成器 - 預測未來是否有利可圖"""
    
    def __init__(self, config):
        self.config = config
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易標籤
        標籤定義: 未來N根K棒內能否獲利
        """
        df = df.copy()
        
        # 計算未來收益
        df = self._calculate_future_returns(df)
        
        # 生成標籤
        df = self._create_labels(df)
        
        return df
    
    def _calculate_future_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算未來不同時間窗口的收益
        """
        # 未來價格
        for horizon in [5, 10, 15, 20]:
            df[f'future_high_{horizon}'] = df['high'].shift(-horizon).rolling(horizon).max()
            df[f'future_low_{horizon}'] = df['low'].shift(-horizon).rolling(horizon).min()
        
        # 計算潛在收益 (考慮止損止盈)
        df['atr'] = df['atr'].fillna(df['close'].pct_change().rolling(14).std() * df['close'])
        
        # 做多機會
        df['long_potential'] = self._calculate_long_potential(df)
        df['long_risk'] = self._calculate_long_risk(df)
        
        # 做空機會
        df['short_potential'] = self._calculate_short_potential(df)
        df['short_risk'] = self._calculate_short_risk(df)
        
        return df
    
    def _calculate_long_potential(self, df: pd.DataFrame) -> pd.Series:
        """
        計算做多潛在收益
        止盈: 2.5 ATR
        止損: 1.5 ATR
        """
        entry = df['close']
        tp = entry + df['atr'] * self.config.atr_tp_multiplier
        sl = entry - df['atr'] * self.config.atr_sl_multiplier
        
        # 檢查未來20根K棒
        future_high = df['future_high_20']
        future_low = df['future_low_20']
        
        # 先止盈還是先止損
        hit_tp = future_high >= tp
        hit_sl = future_low <= sl
        
        # 計算收益
        potential = pd.Series(0.0, index=df.index)
        potential[hit_tp & ~hit_sl] = self.config.atr_tp_multiplier  # 止盈
        potential[hit_sl & ~hit_tp] = -self.config.atr_sl_multiplier  # 止損
        potential[hit_tp & hit_sl] = -self.config.atr_sl_multiplier  # 都觸發,保守假設先止損
        
        return potential
    
    def _calculate_long_risk(self, df: pd.DataFrame) -> pd.Series:
        """
        計算做多風險 (最大回撤)
        """
        entry = df['close']
        future_low = df['future_low_20']
        
        drawdown = (future_low - entry) / df['atr']
        return drawdown.abs()
    
    def _calculate_short_potential(self, df: pd.DataFrame) -> pd.Series:
        """
        計算做空潛在收益
        """
        entry = df['close']
        tp = entry - df['atr'] * self.config.atr_tp_multiplier
        sl = entry + df['atr'] * self.config.atr_sl_multiplier
        
        future_high = df['future_high_20']
        future_low = df['future_low_20']
        
        hit_tp = future_low <= tp
        hit_sl = future_high >= sl
        
        potential = pd.Series(0.0, index=df.index)
        potential[hit_tp & ~hit_sl] = self.config.atr_tp_multiplier
        potential[hit_sl & ~hit_tp] = -self.config.atr_sl_multiplier
        potential[hit_tp & hit_sl] = -self.config.atr_sl_multiplier
        
        return potential
    
    def _calculate_short_risk(self, df: pd.DataFrame) -> pd.Series:
        """
        計算做空風險
        """
        entry = df['close']
        future_high = df['future_high_20']
        
        drawup = (future_high - entry) / df['atr']
        return drawup.abs()
    
    def _create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        創建訓練標籤
        label=1: 值得交易 (預期盈虧比>2)
        label=0: 不值得交易
        """
        # 做多標籤
        df['label_long'] = (
            (df['long_potential'] > 1.5) &  # 至少1.5 ATR收益
            (df['long_potential'] / df['long_risk'].clip(lower=0.1) > 1.5)  # 盈虧比>1.5
        ).astype(int)
        
        # 做空標籤
        df['label_short'] = (
            (df['short_potential'] > 1.5) &
            (df['short_potential'] / df['short_risk'].clip(lower=0.1) > 1.5)
        ).astype(int)
        
        # 合併標籤 (只要有一個方向機會就是1)
        df['label'] = ((df['label_long'] == 1) | (df['label_short'] == 1)).astype(int)
        
        # 方向標籤
        df['label_direction'] = 0  # 0=無, 1=long, -1=short
        df.loc[df['label_long'] == 1, 'label_direction'] = 1
        df.loc[df['label_short'] == 1, 'label_direction'] = -1
        
        return df
    
    def get_label_statistics(self, df: pd.DataFrame) -> dict:
        """
        獲取標籤統計信息
        """
        total = len(df.dropna(subset=['label']))
        positive = df['label'].sum()
        long_labels = df['label_long'].sum()
        short_labels = df['label_short'].sum()
        
        return {
            'total_samples': total,
            'positive_labels': int(positive),
            'positive_rate': float(positive / total * 100) if total > 0 else 0,
            'long_labels': int(long_labels),
            'short_labels': int(short_labels),
            'avg_long_potential': float(df[df['label_long']==1]['long_potential'].mean()) if long_labels > 0 else 0,
            'avg_short_potential': float(df[df['label_short']==1]['short_potential'].mean()) if short_labels > 0 else 0
        }
