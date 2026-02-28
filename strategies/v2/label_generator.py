"""
V2 Label Generator
V2標籤生成器 - BB反轉識別
"""
import pandas as pd
import numpy as np
import ta

class LabelGenerator:
    def __init__(self, config):
        self.config = config
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成BB反轉標籤
        """
        df = df.copy()
        
        # 1. 計算BB和ATR
        df = self._calculate_bb_atr(df)
        
        # 2. 檢測BB觸碰
        df = self._detect_bb_touch(df)
        
        # 3. 識別有效反轉
        df = self._identify_valid_reversals(df)
        
        # 4. 生成標籤
        df['label'] = 0  # 0=無效反轉, 1=有效反轉
        df.loc[df['valid_reversal'], 'label'] = 1
        
        return df
    
    def _calculate_bb_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算Bollinger Bands和ATR
        """
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(
            close=df['close'],
            window=self.config.bb_window,
            window_dev=self.config.bb_std
        )
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        df['atr'] = ta.volatility.average_true_range(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.config.atr_window
        )
        
        return df
    
    def _detect_bb_touch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        檢測BB觸碰
        """
        # 觸碰上軌: 最高價碰到上軌,但收盤價在上軌下方
        df['touch_upper'] = (df['high'] >= df['bb_upper']) & (df['close'] < df['bb_upper'])
        
        # 觸碰下軌: 最低價碰到下軌,但收盤價在下軌上方
        df['touch_lower'] = (df['low'] <= df['bb_lower']) & (df['close'] > df['bb_lower'])
        
        # 觸碰力道 (以ATR為單位)
        df['touch_strength_upper'] = np.where(
            df['touch_upper'],
            (df['high'] - df['bb_upper']) / df['atr'],
            0
        )
        
        df['touch_strength_lower'] = np.where(
            df['touch_lower'],
            (df['bb_lower'] - df['low']) / df['atr'],
            0
        )
        
        return df
    
    def _identify_valid_reversals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        識別有效反轉
        
        有效反轉條件:
        1. 必須有明顯反轉 (至少min_reversal_atr倍ATR)
        2. 反轉後不能突破BB通道 (允許小幅度突破)
        3. 至少反轉到中軌
        """
        lookforward = self.config.reversal_lookforward
        min_reversal = self.config.min_reversal_atr
        tolerance = self.config.breakout_tolerance
        
        df['valid_reversal'] = False
        
        for i in range(len(df) - lookforward):
            current_atr = df.iloc[i]['atr']
            
            # 檢查上軌觸碰後的反轉 (預期下跌)
            if df.iloc[i]['touch_upper']:
                current_high = df.iloc[i]['high']
                current_close = df.iloc[i]['close']
                current_bb_upper = df.iloc[i]['bb_upper']
                current_bb_middle = df.iloc[i]['bb_middle']
                
                # 未來10根K棒的最低價
                future_low = df.iloc[i+1:i+1+lookforward]['low'].min()
                
                # 未來10根K棒的最高價
                future_high = df.iloc[i+1:i+1+lookforward]['high'].max()
                
                # 條件1: 有明顯下跌
                drop_amount = current_close - future_low
                has_drop = drop_amount >= min_reversal * current_atr
                
                # 條件2: 沒有大幅突破上軌
                no_breakout = future_high <= current_bb_upper * (1 + tolerance)
                
                # 條件3: 至少反轉到中軌
                reach_middle = future_low <= current_bb_middle
                
                if has_drop and no_breakout and reach_middle:
                    df.loc[df.index[i], 'valid_reversal'] = True
                    df.loc[df.index[i], 'reversal_direction'] = 'SHORT'
                    df.loc[df.index[i], 'reversal_amount'] = drop_amount / current_atr
            
            # 檢查下軌觸碰後的反轉 (預期上漨)
            elif df.iloc[i]['touch_lower']:
                current_low = df.iloc[i]['low']
                current_close = df.iloc[i]['close']
                current_bb_lower = df.iloc[i]['bb_lower']
                current_bb_middle = df.iloc[i]['bb_middle']
                
                future_high = df.iloc[i+1:i+1+lookforward]['high'].max()
                future_low = df.iloc[i+1:i+1+lookforward]['low'].min()
                
                # 條件1: 有明顯上漨
                rise_amount = future_high - current_close
                has_rise = rise_amount >= min_reversal * current_atr
                
                # 條件2: 沒有大幅突破下軌
                no_breakout = future_low >= current_bb_lower * (1 - tolerance)
                
                # 條件3: 至少反轉到中軌
                reach_middle = future_high >= current_bb_middle
                
                if has_rise and no_breakout and reach_middle:
                    df.loc[df.index[i], 'valid_reversal'] = True
                    df.loc[df.index[i], 'reversal_direction'] = 'LONG'
                    df.loc[df.index[i], 'reversal_amount'] = rise_amount / current_atr
        
        return df
    
    def get_statistics(self, df: pd.DataFrame) -> dict:
        """
        獲取標籤統計資訊
        """
        total_samples = len(df)
        touch_upper = df['touch_upper'].sum()
        touch_lower = df['touch_lower'].sum()
        valid_reversals = df['valid_reversal'].sum()
        
        reversal_long = df[df['reversal_direction'] == 'LONG'].shape[0] if 'reversal_direction' in df.columns else 0
        reversal_short = df[df['reversal_direction'] == 'SHORT'].shape[0] if 'reversal_direction' in df.columns else 0
        
        return {
            'total_samples': total_samples,
            'touch_upper': int(touch_upper),
            'touch_lower': int(touch_lower),
            'total_touches': int(touch_upper + touch_lower),
            'valid_reversals': int(valid_reversals),
            'reversal_rate': float(valid_reversals / (touch_upper + touch_lower) * 100) if (touch_upper + touch_lower) > 0 else 0,
            'reversal_long': int(reversal_long),
            'reversal_short': int(reversal_short)
        }
