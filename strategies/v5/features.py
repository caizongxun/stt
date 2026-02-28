"""
V5 Feature Engineering - Enhanced
V5特徵工程 - 增強版
"""
import pandas as pd
import numpy as np
import talib

class V5FeatureEngine:
    """
    增強版特徵工程
    添加: 型態識別 + 支撓壓力 + 成交量突變
    """
    
    def __init__(self, config):
        self.config = config
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        print("[V5 Features Enhanced]")
        
        if self.config.use_price_features:
            df = self._add_price_features(df)
            print("  + Price")
        
        if self.config.use_volume_features:
            df = self._add_volume_features(df)
            print("  + Volume")
        
        if self.config.use_volatility_features:
            df = self._add_volatility_features(df)
            print("  + Volatility")
        
        if self.config.use_momentum_features:
            df = self._add_momentum_features(df)
            print("  + Momentum")
        
        df = self._add_pattern_features(df)
        print("  + Patterns")
        
        df = self._add_support_resistance(df)
        print("  + Support/Resistance")
        
        df = self._add_volume_surge(df)
        print("  + Volume Surge")
        
        return df
    
    def _add_price_features(self, df):
        for window in self.config.lookback_windows:
            df[f'return_{window}'] = df['close'].pct_change(window)
            ma = df['close'].rolling(window).mean()
            df[f'dist_ma_{window}'] = (df['close'] - ma) / ma
            roll_min = df['low'].rolling(window).min()
            roll_max = df['high'].rolling(window).max()
            df[f'price_position_{window}'] = (df['close'] - roll_min) / (roll_max - roll_min + 1e-10)
        
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
        return df
    
    def _add_volume_features(self, df):
        for window in self.config.lookback_windows:
            vol_ma = df['volume'].rolling(window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / (vol_ma + 1e-10)
            df[f'volume_trend_{window}'] = df['volume'].pct_change(window)
        df['price_volume_corr'] = df['close'].pct_change().rolling(20).corr(df['volume'].pct_change())
        return df
    
    def _add_volatility_features(self, df):
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['atr_pct'] = df['atr'] / df['close']
        for window in [10, 20]:
            df[f'volatility_{window}'] = df['close'].pct_change().rolling(window).std()
            df[f'range_{window}'] = (df['high'].rolling(window).max() - df['low'].rolling(window).min()) / df['close']
        
        bb_period, bb_std = 20, 2
        ma = df['close'].rolling(bb_period).mean()
        std = df['close'].rolling(bb_period).std()
        df['bb_upper'] = ma + bb_std * std
        df['bb_lower'] = ma - bb_std * std
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / ma
        return df
    
    def _add_momentum_features(self, df):
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_fast'] = talib.RSI(df['close'], timeperiod=7)
        macd, signal, hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
        for window in [5, 10, 20]:
            df[f'momentum_{window}'] = df['close'] / df['close'].shift(window) - 1
            df[f'momentum_accel_{window}'] = df[f'momentum_{window}'].diff()
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        return df
    
    def _add_pattern_features(self, df):
        body = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        total_range = df['high'] - df['low']
        
        df['is_doji'] = ((body / (total_range + 1e-10)) < 0.1).astype(int)
        df['is_hammer'] = ((lower_shadow > body * 2) & (upper_shadow < body * 0.3) & (df['close'] > df['open'])).astype(int)
        df['is_inv_hammer'] = ((upper_shadow > body * 2) & (lower_shadow < body * 0.3) & (df['close'] < df['open'])).astype(int)
        
        prev_body = abs(df['close'].shift(1) - df['open'].shift(1))
        df['is_bullish_engulf'] = ((df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1)) & (body > prev_body * 1.2)).astype(int)
        df['is_bearish_engulf'] = ((df['close'] < df['open']) & (df['close'].shift(1) > df['open'].shift(1)) & (body > prev_body * 1.2)).astype(int)
        
        df['consecutive_up'] = (df['close'] > df['close'].shift(1)).astype(int).rolling(3).sum()
        df['consecutive_down'] = (df['close'] < df['close'].shift(1)).astype(int).rolling(3).sum()
        return df
    
    def _add_support_resistance(self, df):
        for window in [10, 20, 40]:
            support = df['low'].rolling(window).min()
            resistance = df['high'].rolling(window).max()
            df[f'dist_support_{window}'] = (df['close'] - support) / df['close']
            df[f'dist_resistance_{window}'] = (resistance - df['close']) / df['close']
            df[f'sr_position_{window}'] = (df['close'] - support) / (resistance - support + 1e-10)
        return df
    
    def _add_volume_surge(self, df):
        vol_ma_20 = df['volume'].rolling(20).mean()
        vol_std_20 = df['volume'].rolling(20).std()
        df['volume_surge'] = (df['volume'] > vol_ma_20 + 2 * vol_std_20).astype(int)
        df['volume_extreme_ratio'] = df['volume'] / (vol_ma_20 + 1e-10)
        
        vol_sma_5 = df['volume'].rolling(5).mean()
        vol_sma_20 = df['volume'].rolling(20).mean()
        df['volume_trend_strength'] = (vol_sma_5 - vol_sma_20) / (vol_sma_20 + 1e-10)
        
        price_change = df['close'].pct_change()
        volume_change = df['volume'].pct_change()
        df['price_volume_sync'] = ((price_change > 0) & (volume_change > 0)).astype(int)
        return df
    
    def get_feature_names(self, df):
        exclude = ['open', 'high', 'low', 'close', 'volume', 'open_time', 'close_time',
                   'bb_upper', 'bb_lower', 'macd_signal',
                   'future_high', 'future_low', 'future_close',
                   'long_return', 'short_return', 'long_drawdown', 'short_drawdown',
                   'label_long', 'label_short', 'label', 'label_binary', 'signal_direction']
        
        feature_names = []
        for col in df.columns:
            if col in exclude or pd.api.types.is_datetime64_any_dtype(df[col]) or df[col].dtype == 'object':
                continue
            if 'future' in col.lower() or 'label' in col.lower():
                continue
            feature_names.append(col)
        
        valid_features = [col for col in feature_names if df[col].notna().sum() > len(df) * 0.5]
        print(f"[V5] Features: {len(valid_features)}")
        return valid_features
