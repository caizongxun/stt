"""
V5 Feature Engineering
V5特徵工程 - 價格導向
"""
import pandas as pd
import numpy as np
import talib

class V5FeatureEngine:
    """
    純價格預測特徵
    不用複雜指標,只用最有效的
    """
    
    def __init__(self, config):
        self.config = config
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成所有特徵"""
        df = df.copy()
        
        print("[V5 Features]")
        
        # 1. 價格特徵
        if self.config.use_price_features:
            df = self._add_price_features(df)
            print("  + Price features")
        
        # 2. 成交量特徵
        if self.config.use_volume_features:
            df = self._add_volume_features(df)
            print("  + Volume features")
        
        # 3. 波動率特徵
        if self.config.use_volatility_features:
            df = self._add_volatility_features(df)
            print("  + Volatility features")
        
        # 4. 動量特徵
        if self.config.use_momentum_features:
            df = self._add_momentum_features(df)
            print("  + Momentum features")
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """價格特徵"""
        for window in self.config.lookback_windows:
            # 報酬率
            df[f'return_{window}'] = df['close'].pct_change(window)
            
            # 距離均線
            ma = df['close'].rolling(window).mean()
            df[f'dist_ma_{window}'] = (df['close'] - ma) / ma
            
            # 價格位置(區間百分比)
            roll_min = df['low'].rolling(window).min()
            roll_max = df['high'].rolling(window).max()
            df[f'price_position_{window}'] = (df['close'] - roll_min) / (roll_max - roll_min + 1e-10)
        
        # K線實體大小
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """成交量特徵"""
        for window in self.config.lookback_windows:
            # 成交量比率
            vol_ma = df['volume'].rolling(window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / (vol_ma + 1e-10)
            
            # 成交量趨勢
            df[f'volume_trend_{window}'] = df['volume'].pct_change(window)
        
        # 價量關係
        df['price_volume_corr'] = df['close'].pct_change().rolling(20).corr(df['volume'].pct_change())
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """波動率特徵"""
        # ATR
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['atr_pct'] = df['atr'] / df['close']
        
        for window in [10, 20]:
            # 波動率
            df[f'volatility_{window}'] = df['close'].pct_change().rolling(window).std()
            
            # 高低幅度
            df[f'range_{window}'] = (df['high'].rolling(window).max() - df['low'].rolling(window).min()) / df['close']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        ma = df['close'].rolling(bb_period).mean()
        std = df['close'].rolling(bb_period).std()
        df['bb_upper'] = ma + bb_std * std
        df['bb_lower'] = ma - bb_std * std
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / ma
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """動量特徵"""
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_fast'] = talib.RSI(df['close'], timeperiod=7)
        
        # MACD
        macd, signal, hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
        
        # 動量指標
        for window in [5, 10, 20]:
            df[f'momentum_{window}'] = df['close'] / df['close'].shift(window) - 1
            df[f'momentum_accel_{window}'] = df[f'momentum_{window}'].diff()
        
        # ADX (趨勢強度)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> list:
        """獲取特徵名稱"""
        # 排除原始列
        exclude = ['open', 'high', 'low', 'close', 'volume', 'open_time', 
                   'bb_upper', 'bb_lower', 'macd_signal']
        
        feature_names = [col for col in df.columns if col not in exclude]
        
        # 只保留有效特徵(非null)
        valid_features = []
        for col in feature_names:
            if df[col].notna().sum() > len(df) * 0.5:  # 至少50%有效
                valid_features.append(col)
        
        print(f"[V5] Total features: {len(valid_features)}")
        return valid_features
