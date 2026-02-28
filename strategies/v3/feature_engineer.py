"""
V3 Feature Engineer
V3特徵工程 - 50+特徵
"""
import pandas as pd
import numpy as np
import ta

class FeatureEngineer:
    """擴展特徵工程"""
    
    def __init__(self, config):
        self.config = config
    
    def engineer(self, df: pd.DataFrame) -> tuple:
        """
        生成所有特徵
        返回: (df, feature_names)
        """
        df = df.copy()
        feature_names = []
        
        # 1. 基本价格特徵
        df, price_features = self._price_features(df)
        feature_names.extend(price_features)
        
        # 2. 技術指标特徵
        df, tech_features = self._technical_features(df)
        feature_names.extend(tech_features)
        
        # 3. 波动率特徵
        df, vol_features = self._volatility_features(df)
        feature_names.extend(vol_features)
        
        # 4. 成交量特徵
        df, volume_features = self._volume_features(df)
        feature_names.extend(volume_features)
        
        # 5. 动量特徵
        df, momentum_features = self._momentum_features(df)
        feature_names.extend(momentum_features)
        
        # 6. 结构特徵
        df, structure_features = self._structure_features(df)
        feature_names.extend(structure_features)
        
        # 7. 时间特徵
        df, time_features = self._time_features(df)
        feature_names.extend(time_features)
        
        # 8. 信号强度特徵
        if 'signal_long_strength' in df.columns:
            feature_names.extend(['signal_long_strength', 'signal_short_strength'])
        
        return df, feature_names
    
    def _price_features(self, df: pd.DataFrame) -> tuple:
        """价格相关特徵"""
        features = []
        
        # 价格位置
        df['price_position_5'] = (df['close'] - df['low'].rolling(5).min()) / \
                                  (df['high'].rolling(5).max() - df['low'].rolling(5).min())
        df['price_position_20'] = (df['close'] - df['low'].rolling(20).min()) / \
                                   (df['high'].rolling(20).max() - df['low'].rolling(20).min())
        features.extend(['price_position_5', 'price_position_20'])
        
        # K线形态
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
        features.extend(['body_size', 'upper_shadow', 'lower_shadow'])
        
        # 价格变化
        for period in [1, 3, 5, 10]:
            df[f'return_{period}'] = df['close'].pct_change(period)
            features.append(f'return_{period}')
        
        return df, features
    
    def _technical_features(self, df: pd.DataFrame) -> tuple:
        """技术指标特徵"""
        features = []
        
        # RSI
        for window in [7, 14, 21]:
            df[f'rsi_{window}'] = ta.momentum.rsi(df['close'], window=window)
            features.append(f'rsi_{window}')
        
        # MACD
        features.extend(['macd', 'macd_signal', 'macd_diff'])
        
        # ADX
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        features.append('adx')
        
        # CCI
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
        features.append('cci')
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        features.extend(['stoch_k', 'stoch_d'])
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        features.append('williams_r')
        
        return df, features
    
    def _volatility_features(self, df: pd.DataFrame) -> tuple:
        """波动率特徵"""
        features = []
        
        # ATR相关
        df['atr_pct'] = df['atr'] / df['close']
        df['atr_ratio'] = df['atr'] / df['atr'].rolling(20).mean()
        features.extend(['atr', 'atr_pct', 'atr_ratio'])
        
        # BB相关
        features.extend(['bb_width', 'bb_position'])
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).quantile(0.25)).astype(int)
        features.append('bb_squeeze')
        
        # 波动率
        df['volatility_5'] = df['close'].pct_change().rolling(5).std()
        df['volatility_20'] = df['close'].pct_change().rolling(20).std()
        features.extend(['volatility_5', 'volatility_20'])
        
        # 真实波动率
        df['realized_vol'] = np.sqrt(((df['high'] / df['low']).apply(np.log) ** 2).rolling(10).mean())
        features.append('realized_vol')
        
        return df, features
    
    def _volume_features(self, df: pd.DataFrame) -> tuple:
        """成交量特徵"""
        features = []
        
        # 成交量比率
        features.append('volume_ratio')
        
        # OBV
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['obv_slope'] = df['obv'].pct_change(5)
        features.extend(['obv', 'obv_slope'])
        
        # 价量相关
        df['volume_price_corr'] = df['close'].rolling(20).corr(df['volume'])
        features.append('volume_price_corr')
        
        # 成交量突增
        df['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)
        features.append('volume_spike')
        
        return df, features
    
    def _momentum_features(self, df: pd.DataFrame) -> tuple:
        """动量特徵"""
        features = []
        
        # 动量
        features.extend(['momentum', 'momentum_strength'])
        
        # ROC
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(period) * 100
            features.append(f'roc_{period}')
        
        # 动量指标
        df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
        features.append('mfi')
        
        return df, features
    
    def _structure_features(self, df: pd.DataFrame) -> tuple:
        """结构特徵"""
        features = []
        
        # 移动平均位置
        df['ma_position'] = (df['close'] - df['ma_fast']) / df['ma_fast']
        df['ma_distance'] = (df['ma_fast'] - df['ma_slow']) / df['ma_slow']
        features.extend(['ma_position', 'ma_distance'])
        
        # 支撑阻力距离
        df['dist_to_high_20'] = (df['high'].rolling(20).max() - df['close']) / df['close']
        df['dist_to_low_20'] = (df['close'] - df['low'].rolling(20).min()) / df['close']
        features.extend(['dist_to_high_20', 'dist_to_low_20'])
        
        return df, features
    
    def _time_features(self, df: pd.DataFrame) -> tuple:
        """时间特徵"""
        features = []
        
        if 'open_time' in df.columns:
            df['hour'] = pd.to_datetime(df['open_time']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['open_time']).dt.dayofweek
            
            # 时间段编码
            df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['is_europe_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['is_us_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
            
            features.extend(['hour', 'day_of_week', 'is_asian_session', 'is_europe_session', 'is_us_session'])
        
        return df, features
