"""
V6 Label Generation - Reversal Targets
"""
import numpy as np
import pandas as pd


class V6LabelGenerator:
    def __init__(self, config):
        self.config = config

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["future_return"] = df["close"].shift(-self.config.forward_bars) / df[
            "close"
        ] - 1

        df["label_long"] = (
            (df["future_return"] >= self.config.reversal_target_pct)
            & (df["rsi"] < self.config.oversold_rsi)
            & (df["bb_position"] <= self.config.min_bb_position)
            & (df["zscore_20"] < -0.8)
        ).astype(int)

        df["label_short"] = (
            (df["future_return"] <= -self.config.reversal_target_pct)
            & (df["rsi"] > self.config.overbought_rsi)
            & (df["bb_position"] >= 1 - self.config.min_bb_position)
            & (df["zscore_20"] > 0.8)
        ).astype(int)

        df["label"] = 0
        df.loc[df["label_long"] == 1, "label"] = 1
        df.loc[df["label_short"] == 1, "label"] = -1
        df["label_binary"] = (df["label"] != 0).astype(int)

        self._print_statistics(df)
        return df

    def _print_statistics(self, df: pd.DataFrame):
        valid = df["label_binary"].notna()
        total = valid.sum()
        if total == 0:
            print("[V6] 無有效標籤")
            return

        positive = (df.loc[valid, "label_binary"] == 1).sum()
        pos_rate = positive / total * 100

        long_cnt = (df["label"] == 1).sum()
        short_cnt = (df["label"] == -1).sum()

        print(f"[V6] 樣本: {total}, 正樣本: {positive} ({pos_rate:.1f}%)")
        print(f"[V6] Long: {long_cnt}, Short: {short_cnt}")
