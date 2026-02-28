"""
V6 Feature Engineering - Reversal Focused
"""
import numpy as np
import pandas as pd
import talib


class V6FeatureEngineer:
    def __init__(self, config):
        self.config = config

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["return_1"] = df["close"].pct_change()
        df["return_3"] = df["close"].pct_change(3)
        df["return_6"] = df["close"].pct_change(6)

        df["atr"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)

        ma20 = df["close"].rolling(20).mean()
        std20 = df["close"].rolling(20).std()
        df["bb_mid"] = ma20
        df["bb_upper"] = ma20 + self.config.bb_std * std20
        df["bb_lower"] = ma20 - self.config.bb_std * std20
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (ma20 + 1e-9)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"] + 1e-9
        )
        df["zscore_20"] = (df["close"] - ma20) / (std20 + 1e-9)

        df["rsi"] = talib.RSI(df["close"], timeperiod=14)
        df["rsi_fast"] = talib.RSI(df["close"], timeperiod=7)
        df["rsi_slope"] = df["rsi"].diff()

        df["ema_10"] = talib.EMA(df["close"], timeperiod=10)
        df["ema_30"] = talib.EMA(df["close"], timeperiod=30)
        df["ema_gap"] = (df["ema_10"] - df["ema_30"]) / (df["ema_30"] + 1e-9)

        df["upper_wick"] = (
            df["high"] - df[["open", "close"]].max(axis=1)
        ) / df["open"]
        df["lower_wick"] = (
            df[["open", "close"]].min(axis=1) - df["low"]
        ) / df["open"]
        df["body_pct"] = (df["close"] - df["open"]) / df["open"]

        df["volume_ratio"] = df["volume"] / (
            df["volume"].rolling(20).mean() + 1e-9
        )
        df["vol_chg"] = df["volume"].pct_change()

        return df

    def get_feature_names(self, df: pd.DataFrame):
        exclude = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_time",
            "close_time",
            "future_return",
            "label",
            "label_long",
            "label_short",
            "label_binary",
        }

        feature_names = []
        for col in df.columns:
            if col in exclude:
                continue
            if pd.api.types.is_datetime64_any_dtype(df[col]) or df[col].dtype == "object":
                continue
            if "future" in col or "label" in col:
                continue
            feature_names.append(col)

        return [
            col
            for col in feature_names
            if df[col].notna().sum() >= len(df) * 0.6
        ]
