"""
V6 Backtester - Reversal Execution
"""
import numpy as np
import pandas as pd

from .features import V6FeatureEngineer


class V6Backtester:
    def __init__(self, config):
        self.config = config
        self.trades = []
        self.equity_curve = []
        self.current_capital = config.capital

    def run(self, long_model, short_model, df, feature_names):
        df = self._prepare_data(long_model, short_model, df, feature_names)
        self._simulate(df)

        if not self.trades:
            return {"status": "no_trades"}

        return self._results(df)

    def _prepare_data(self, long_model, short_model, df, feature_names):
        feature_engine = V6FeatureEngineer(self.config)
        df = feature_engine.generate(df.copy())

        X = df[feature_names].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())

        df["long_proba"] = (
            long_model.predict_proba(X)[:, 1] if long_model is not None else 0
        )
        df["short_proba"] = (
            short_model.predict_proba(X)[:, 1] if short_model is not None else 0
        )
        df["long_signal"] = (df["long_proba"] >= self.config.long_threshold).astype(int)
        df["short_signal"] = (
            df["short_proba"] >= self.config.short_threshold
        ).astype(int)
        return df

    def _simulate(self, df: pd.DataFrame):
        position = None

        for idx, row in df.iterrows():
            if position:
                exit_price, reason = self._check_exit(position, row, idx)
                if exit_price is not None:
                    self._close(position, exit_price, reason, row.get("open_time", idx))
                    position = None

            if position is None:
                if row["long_signal"] and not row["short_signal"]:
                    position = self._open(row, idx, "LONG")
                elif row["short_signal"]:
                    position = self._open(row, idx, "SHORT")

            self.equity_curve.append(
                {
                    "bar": idx,
                    "timestamp": row.get("open_time", idx),
                    "equity": self.current_capital,
                }
            )

    def _open(self, row, idx, direction):
        price = row["close"]
        atr = row.get("atr", price * 0.015)
        if pd.isna(atr) or atr <= 0:
            atr = price * 0.015

        if direction == "LONG":
            stop = price - atr * self.config.atr_sl_multiplier
            take = price + atr * self.config.atr_tp_multiplier
        else:
            stop = price + atr * self.config.atr_sl_multiplier
            take = price - atr * self.config.atr_tp_multiplier

        base = self.current_capital if self.config.use_compound else self.config.capital
        position_value = base * self.config.position_pct * self.config.leverage

        return {
            "direction": direction,
            "entry_price": price,
            "entry_bar": idx,
            "stop": stop,
            "take": take,
            "position_value": position_value,
        }

    def _check_exit(self, position, row, idx):
        price = row["close"]
        if position["direction"] == "LONG":
            if row["low"] <= position["stop"]:
                return position["stop"], "STOP"
            if row["high"] >= position["take"]:
                return position["take"], "TP"
            if row["short_signal"]:
                return price, "REVERSE"
        else:
            if row["high"] >= position["stop"]:
                return position["stop"], "STOP"
            if row["low"] <= position["take"]:
                return position["take"], "TP"
            if row["long_signal"]:
                return price, "REVERSE"

        if idx - position["entry_bar"] >= self.config.max_hold_bars:
            return price, "TIME"
        return None, None

    def _close(self, position, exit_price, reason, exit_time):
        direction = position["direction"]
        change = (
            (exit_price - position["entry_price"]) / position["entry_price"]
            if direction == "LONG"
            else (position["entry_price"] - exit_price) / position["entry_price"]
        )

        pnl = position["position_value"] * change
        cost = (
            position["position_value"]
            * (self.config.fee_rate + self.config.slippage)
            * 2
        )
        pnl -= cost

        self.current_capital += pnl
        self.trades.append(
            {
                "direction": direction,
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pct": pnl / position["position_value"] * 100,
                "exit_reason": reason,
                "exit_time": exit_time,
            }
        )

    def _results(self, df):
        trades_df = pd.DataFrame(self.trades)
        winning = trades_df[trades_df["pnl"] > 0]
        losing = trades_df[trades_df["pnl"] <= 0]

        win_rate = winning.shape[0] / len(trades_df)
        profit_factor = (
            abs(winning["pnl"].sum() / losing["pnl"].sum())
            if len(losing) > 0 and losing["pnl"].sum() != 0
            else float("inf")
        )

        equity_df = pd.DataFrame(self.equity_curve)
        equity_df["peak"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = (equity_df["peak"] - equity_df["equity"]) / equity_df[
            "peak"
        ]

        days = (
            (df.iloc[-1]["open_time"] - df.iloc[0]["open_time"]).days
            if "open_time" in df.columns
            else len(df) / 96
        )
        total_return = (
            self.current_capital - self.config.capital
        ) / self.config.capital * 100

        return {
            "status": "success",
            "capital": {
                "initial": float(self.config.capital),
                "final": float(self.current_capital),
                "total_return_pct": float(total_return),
                "max_drawdown_pct": float(equity_df["drawdown"].max() * 100),
            },
            "trades": {
                "total": int(len(trades_df)),
                "win_rate_pct": float(win_rate * 100),
                "profit_factor": float(profit_factor),
                "avg_win": float(winning["pnl"].mean() if len(winning) else 0),
                "avg_loss": float(losing["pnl"].mean() if len(losing) else 0),
            },
            "exit_reasons": trades_df["exit_reason"].value_counts().to_dict(),
            "sample_trades": trades_df.tail(10).to_dict("records"),
        }
