"""
V2 Feature Engineer
V2特徵工程
"""
import pandas as pd
import numpy as np
import ta

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
    
    def engineer(self, df: pd.DataFrame) -> tuple:
        """
        生成所有特徵
        返回: (df_with_features, feature_names)
        """
        df = df.copy()
        feature_names = []
        
        # 1. BB相關特徵
        df, bb_features = self._add_bb_features(df)
        feature_names.extend(bb_features)
        
        # 2. 技術指標
        if self.config.use_technical_indicators:
            df, tech_features = self._add_technical_indicators(df)
            feature_names.extend(tech_features)
        
        # 3. 市場狀態
        if self.config.use_market_regime:
            df, regime_features = self._add_market_regime(df)
            feature_names.extend(regime_features)
        
        # 4. 歷史成功率
        if self.config.use_historical_success:
            df, hist_features = self._add_historical_success(df)
            feature_names.extend(hist_features)
        
        return df, feature_names
    
    def _add_bb_features(self, df: pd.DataFrame) -> tuple:
        """
        BB相關特徵
        """
        features = []
        
        # 基本BB位置
        features.append('bb_position')
        features.append('bb_width')
        
        # BB斜率
        df['bb_upper_slope'] = (df['bb_upper'] - df['bb_upper'].shift(5)) / df['bb_upper'].shift(5)
        df['bb_lower_slope'] = (df['bb_lower'] - df['bb_lower'].shift(5)) / df['bb_lower'].shift(5)
        df['bb_middle_slope'] = (df['bb_middle'] - df['bb_middle'].shift(5)) / df['bb_middle'].shift(5)
        features.extend(['bb_upper_slope', 'bb_lower_slope', 'bb_middle_slope'])
        
        # 距離BB通道的距離(以ATR為單位)
        df['dist_to_upper'] = (df['bb_upper'] - df['close']) / df['atr']
        df['dist_to_lower'] = (df['close'] - df['bb_lower']) / df['atr']
        df['dist_to_middle'] = abs(df['close'] - df['bb_middle']) / df['atr']
        features.extend(['dist_to_upper', 'dist_to_lower', 'dist_to_middle'])
        
        # BB擠壓狀態
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.8).astype(int)
        features.append('bb_squeeze')
        
        # 觸碰力道
        if 'touch_strength_upper' in df.columns:
            features.append('touch_strength_upper')
        if 'touch_strength_lower' in df.columns:
            features.append('touch_strength_lower')
        
        return df, features
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> tuple:
        """
        技術指標
        """
        features = []
        
        # RSI
        df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
        df['rsi_28'] = ta.momentum.rsi(df['close'], window=28)
        features.extend(['rsi_14', 'rsi_28'])
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        features.extend(['macd', 'macd_signal', 'macd_diff'])
        
        # ADX (趨勢強度)
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        features.append('adx')
        
        # 移動平均
        for period in [20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
            features.extend([f'sma_{period}', f'price_to_sma_{period}'])
        
        # 成交量
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        features.extend(['volume_sma_20', 'volume_ratio'])
        
        return df, features
    
    def _add_market_regime(self, df: pd.DataFrame) -> tuple:
        """
        市場狀態特徵
        """
        features = []
        
        # 趨勢方向
        df['trend_up'] = (df['sma_20'] > df['sma_50']).astype(int)
        features.append('trend_up')
        
        # 波動率狀態
        df['atr_ratio'] = df['atr'] / df['close']
        df['volatility_high'] = (df['atr_ratio'] > df['atr_ratio'].rolling(50).mean() * 1.2).astype(int)
        features.extend(['atr_ratio', 'volatility_high'])
        
        # 成交量異常
        df['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(int)
        features.append('volume_spike')
        
        return df, features
    
    def _add_historical_success(self, df: pd.DataFrame) -> tuple:
        """
        歷史成功率特徵
        """
        features = []
        
        if 'valid_reversal' not in df.columns:
            return df, features
        
        # 過去100根K棒的觸碰次數和成功率
        df['recent_touch_upper'] = df['touch_upper'].rolling(100, min_periods=1).sum()
        df['recent_touch_lower'] = df['touch_lower'].rolling(100, min_periods=1).sum()
        df['recent_success'] = df['valid_reversal'].rolling(100, min_periods=1).sum()
        
        df['success_rate'] = df['recent_success'] / (df['recent_touch_upper'] + df['recent_touch_lower'] + 1)
        features.append('success_rate')
        
        # 時間段成功率 (亞洲/歐洲/美洲時段)
        if 'open_time' in df.columns:
            df['hour'] = pd.to_datetime(df['open_time']).dt.hour
            hour_success = df.groupby('hour')['valid_reversal'].transform('mean')
            df['hour_success_rate'] = hour_success
            features.append('hour_success_rate')
        
        return df, features
