"""
V2 Backtester
V2回測引擎 - ATR風控 + BB反轉
"""
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from pathlib import Path

class Backtester:
    def __init__(self, config):
        self.config = config
        self.trades = []
        self.equity_curve = []
        
    def run(self, model, df: pd.DataFrame, feature_names: list) -> dict:
        """
        運行V2策略回測
        """
        # 1. 準備數據
        df = self._prepare_data(df, model, feature_names)
        
        # 2. 模擬交易
        self._simulate_trades(df)
        
        # 3. 計算結果
        results = self._calculate_results(df)
        
        return results
    
    def _prepare_data(self, df: pd.DataFrame, model, feature_names: list) -> pd.DataFrame:
        """
        準備回測數據
        """
        df = df.copy()
        
        # 計算BB和ATR
        from .label_generator import LabelGenerator
        label_gen = LabelGenerator(self.config)
        df = label_gen._calculate_bb_atr(df)
        df = label_gen._detect_bb_touch(df)
        
        # 生成特徵 - 傳入is_backtest=True
        from .feature_engineer import FeatureEngineer
        feat_eng = FeatureEngineer(self.config)
        df, _ = feat_eng.engineer(df, is_backtest=True)
        
        # 預測
        X = df[feature_names]
        df['pred_proba'] = model.predict_proba(X)[:, 1]
        df['pred'] = (df['pred_proba'] >= self.config.predict_threshold).astype(int)
        
        # 交易信號
        df['signal_long'] = (df['touch_lower']) & (df['pred'] == 1)
        df['signal_short'] = (df['touch_upper']) & (df['pred'] == 1)
        
        return df
    
    def _simulate_trades(self, df: pd.DataFrame):
        """
        模擬交易執行
        """
        capital = self.config.capital
        position = None
        daily_trades = 0
        last_trade_date = None
        
        for i in range(len(df)):
            row = df.iloc[i]
            current_date = pd.to_datetime(row['open_time']).date() if 'open_time' in df.columns else None
            
            # 重置每日交易數
            if current_date != last_trade_date:
                daily_trades = 0
                last_trade_date = current_date
            
            # 檢查現有仓位
            if position is not None:
                exit_result = self._check_exit(position, row, i)
                if exit_result:
                    # 平倉
                    pnl = exit_result['pnl']
                    capital += pnl
                    
                    self.trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': row['open_time'] if 'open_time' in df.columns else i,
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_result['exit_price'],
                        'exit_reason': exit_result['reason'],
                        'size': position['size'],
                        'pnl': pnl,
                        'pnl_pct': pnl / position['size'],
                        'capital': capital,
                        'bars_held': i - position['entry_idx']
                    })
                    
                    position = None
            
            # 檢查開倉信號
            if position is None and daily_trades < self.config.max_trades_per_day:
                if row['signal_long']:
                    position = self._open_position(
                        row, 'LONG', capital, i
                    )
                    daily_trades += 1
                
                elif row['signal_short']:
                    position = self._open_position(
                        row, 'SHORT', capital, i
                    )
                    daily_trades += 1
            
            # 記錄權益曲線
            self.equity_curve.append({
                'time': row['open_time'] if 'open_time' in df.columns else i,
                'capital': capital,
                'position': position['direction'] if position else 'NONE'
            })
    
    def _open_position(self, row, direction: str, capital: float, idx: int) -> dict:
        """
        開倉
        """
        entry_price = row['close']
        atr = row['atr']
        
        # 計算止損止盈
        if direction == 'LONG':
            stop_loss = entry_price - (atr * self.config.atr_sl_multiplier)
            take_profit = entry_price + (atr * self.config.atr_tp_multiplier)
        else:  # SHORT
            stop_loss = entry_price + (atr * self.config.atr_sl_multiplier)
            take_profit = entry_price - (atr * self.config.atr_tp_multiplier)
        
        # 計算仓位大小
        risk_amount = capital * self.config.max_risk_per_trade
        stop_distance = abs(entry_price - stop_loss)
        size = risk_amount / stop_distance if stop_distance > 0 else capital * 0.1
        size = min(size, capital * self.config.initial_position_pct)  # 不超過15%
        
        return {
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': row['open_time'] if 'open_time' in row else idx,
            'entry_idx': idx,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'size': size,
            'atr': atr
        }
    
    def _check_exit(self, position: dict, row, idx: int) -> dict:
        """
        檢查是否需要平倉
        """
        high = row['high']
        low = row['low']
        close = row['close']
        
        if position['direction'] == 'LONG':
            # 止損
            if low <= position['stop_loss']:
                exit_price = position['stop_loss']
                pnl = (exit_price - position['entry_price']) * position['size'] / position['entry_price']
                pnl -= abs(pnl) * self.config.fee_rate * 2  # 雙邊手續費
                return {'exit_price': exit_price, 'reason': 'STOP_LOSS', 'pnl': pnl}
            
            # 止盈
            if high >= position['take_profit']:
                exit_price = position['take_profit']
                pnl = (exit_price - position['entry_price']) * position['size'] / position['entry_price']
                pnl -= abs(pnl) * self.config.fee_rate * 2
                return {'exit_price': exit_price, 'reason': 'TAKE_PROFIT', 'pnl': pnl}
        
        else:  # SHORT
            # 止損
            if high >= position['stop_loss']:
                exit_price = position['stop_loss']
                pnl = (position['entry_price'] - exit_price) * position['size'] / position['entry_price']
                pnl -= abs(pnl) * self.config.fee_rate * 2
                return {'exit_price': exit_price, 'reason': 'STOP_LOSS', 'pnl': pnl}
            
            # 止盈
            if low <= position['take_profit']:
                exit_price = position['take_profit']
                pnl = (position['entry_price'] - exit_price) * position['size'] / position['entry_price']
                pnl -= abs(pnl) * self.config.fee_rate * 2
                return {'exit_price': exit_price, 'reason': 'TAKE_PROFIT', 'pnl': pnl}
        
        return None
    
    def _calculate_results(self, df: pd.DataFrame) -> dict:
        """
        計算回測結果
        """
        if not self.trades:
            return {
                'status': 'no_trades',
                'message': '沒有生成任何交易信號'
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        # 基本統計
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 盈虧統計
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else 0
        
        # 收益率
        initial_capital = self.config.capital
        final_capital = trades_df.iloc[-1]['capital']
        total_return = (final_capital - initial_capital) / initial_capital
        
        # 回撤
        equity_curve = pd.DataFrame(self.equity_curve)
        equity_curve['drawdown'] = equity_curve['capital'] / equity_curve['capital'].cummax() - 1
        max_drawdown = equity_curve['drawdown'].min()
        
        # 持倉時間
        avg_bars = trades_df['bars_held'].mean()
        
        # 方向統計
        long_trades = trades_df[trades_df['direction'] == 'LONG']
        short_trades = trades_df[trades_df['direction'] == 'SHORT']
        
        results = {
            'status': 'success',
            'backtest_period': {
                'start': df.iloc[0]['open_time'] if 'open_time' in df.columns else 0,
                'end': df.iloc[-1]['open_time'] if 'open_time' in df.columns else len(df),
                'total_bars': len(df)
            },
            'capital': {
                'initial': float(initial_capital),
                'final': float(final_capital),
                'total_pnl': float(total_pnl),
                'total_return': float(total_return * 100),
                'max_drawdown': float(max_drawdown * 100)
            },
            'trades': {
                'total': int(total_trades),
                'winning': int(winning_trades),
                'losing': int(losing_trades),
                'win_rate': float(win_rate * 100),
                'avg_bars_held': float(avg_bars)
            },
            'profit': {
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss),
                'profit_factor': float(profit_factor),
                'avg_pnl_per_trade': float(total_pnl / total_trades)
            },
            'direction': {
                'long_trades': int(len(long_trades)),
                'long_win_rate': float(len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) * 100) if len(long_trades) > 0 else 0,
                'short_trades': int(len(short_trades)),
                'short_win_rate': float(len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) * 100) if len(short_trades) > 0 else 0
            },
            'exit_reasons': trades_df['exit_reason'].value_counts().to_dict(),
            'trades_detail': trades_df.tail(10).to_dict('records')  # 最近10筆交易
        }
        
        return results
