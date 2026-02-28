"""
V5 Backtester
V5回測引擎
"""
import pandas as pd
import numpy as np

from .features import V5FeatureEngine

class V5Backtester:
    """
    V5回測引擎 - 智能執行
    """
    
    def __init__(self, config):
        self.config = config
        self.trades = []
        self.positions = []
        self.equity_curve = []
        self.current_capital = config.capital
        self.peak_capital = config.capital
        self.daily_trades = 0
        self.last_trade_bar = -999
    
    def run(self, models, df: pd.DataFrame, feature_names: list) -> dict:
        """執行回測"""
        print("\n[V5 Backtesting]")
        
        df = self._prepare_data(models, df, feature_names)
        print(f"  Bars: {len(df)}")
        print(f"  Signals: {df['trade_signal'].sum()}")
        
        self._simulate_trading(df)
        print(f"  Trades: {len(self.trades)}")
        
        if len(self.trades) == 0:
            return {'status': 'no_trades', 'message': '無交易信號'}
        
        return self._calculate_results(df)
    
    def _prepare_data(self, models, df, feature_names):
        """準備數據"""
        df = df.copy()
        
        # 生成特徵
        feature_engine = V5FeatureEngine(self.config)
        df = feature_engine.generate(df)
        
        # 預測
        X = df[feature_names].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())
        
        probas = [m.predict_proba(X)[:, 1] for m in models]
        df['pred_proba'] = np.mean(probas, axis=0)
        df['trade_signal'] = (df['pred_proba'] >= self.config.predict_threshold).astype(int)
        
        # 方向判斷 - 用價格動量
        df['price_momentum'] = df['close'].pct_change(3)  # 3根K棒動量
        df['signal_direction'] = np.where(df['price_momentum'] > 0, 1, -1)
        
        return df
    
    def _simulate_trading(self, df):
        """模擬交易"""
        for i in range(len(df)):
            row = df.iloc[i]
            
            # 重置每日計數
            if i > 0:
                current_date = pd.to_datetime(row.get('open_time', i)).date() if 'open_time' in df.columns else i // 96
                prev_date = pd.to_datetime(df.iloc[i-1].get('open_time', i-1)).date() if 'open_time' in df.columns else (i-1) // 96
                if current_date != prev_date:
                    self.daily_trades = 0
            
            # 管理持倉
            self._manage_positions(row, i)
            
            # 開倉
            if len(self.positions) < self.config.max_positions and \
               self.daily_trades < self.config.max_trades_per_day and \
               i - self.last_trade_bar >= self.config.min_bars_between:
                
                if row['trade_signal'] == 1:
                    direction = 'LONG' if row['signal_direction'] == 1 else 'SHORT'
                    self._open_position(row, i, direction)
            
            # 記錄權益
            self.equity_curve.append({
                'bar': i,
                'time': row.get('open_time', i),
                'capital': self.current_capital,
                'positions': len(self.positions)
            })
    
    def _open_position(self, row, bar_idx, direction):
        """開倉"""
        entry_price = row['close']
        atr = row['atr'] if 'atr' in row and not pd.isna(row['atr']) else entry_price * 0.02
        
        # 計算止損止盈
        if direction == 'LONG':
            stop_loss = entry_price - atr * self.config.atr_sl_multiplier
            take_profit = entry_price + atr * self.config.atr_tp_multiplier
        else:
            stop_loss = entry_price + atr * self.config.atr_sl_multiplier
            take_profit = entry_price - atr * self.config.atr_tp_multiplier
        
        # 倉位大小
        if self.config.use_compound:
            available_capital = self.current_capital
        else:
            available_capital = self.config.capital
        
        position_value = available_capital * self.config.position_pct * self.config.leverage
        
        position = {
            'direction': direction,
            'entry_bar': bar_idx,
            'entry_price': entry_price,
            'entry_time': row.get('open_time', bar_idx),
            'position_value': position_value,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_stop': None,
            'peak_profit': 0,
            'atr': atr
        }
        
        self.positions.append(position)
        self.daily_trades += 1
        self.last_trade_bar = bar_idx
    
    def _manage_positions(self, row, bar_idx):
        """管理持倉"""
        closed = []
        
        for pos in self.positions:
            # 計算當前盈虧
            if pos['direction'] == 'LONG':
                current_profit = (row['close'] - pos['entry_price']) / pos['entry_price']
            else:
                current_profit = (pos['entry_price'] - row['close']) / pos['entry_price']
            
            # 更新移動止損
            if self.config.use_trailing_stop:
                if current_profit > self.config.trailing_activation:
                    if pos['trailing_stop'] is None:
                        pos['trailing_stop'] = row['close']
                    
                    if pos['direction'] == 'LONG':
                        if row['close'] > pos['trailing_stop']:
                            pos['trailing_stop'] = row['close'] - row['close'] * self.config.trailing_distance
                    else:
                        if row['close'] < pos['trailing_stop']:
                            pos['trailing_stop'] = row['close'] + row['close'] * self.config.trailing_distance
            
            exit_reason = None
            exit_price = None
            
            # 移動止損
            if pos['trailing_stop'] is not None:
                if pos['direction'] == 'LONG' and row['low'] <= pos['trailing_stop']:
                    exit_reason = 'TRAILING_STOP'
                    exit_price = pos['trailing_stop']
                elif pos['direction'] == 'SHORT' and row['high'] >= pos['trailing_stop']:
                    exit_reason = 'TRAILING_STOP'
                    exit_price = pos['trailing_stop']
            
            # 止損
            if exit_reason is None:
                if pos['direction'] == 'LONG' and row['low'] <= pos['stop_loss']:
                    exit_reason = 'STOP_LOSS'
                    exit_price = pos['stop_loss']
                elif pos['direction'] == 'SHORT' and row['high'] >= pos['stop_loss']:
                    exit_reason = 'STOP_LOSS'
                    exit_price = pos['stop_loss']
            
            # 止盈
            if exit_reason is None:
                if pos['direction'] == 'LONG' and row['high'] >= pos['take_profit']:
                    exit_reason = 'TAKE_PROFIT'
                    exit_price = pos['take_profit']
                elif pos['direction'] == 'SHORT' and row['low'] <= pos['take_profit']:
                    exit_reason = 'TAKE_PROFIT'
                    exit_price = pos['take_profit']
            
            # 時間止損
            if exit_reason is None and bar_idx - pos['entry_bar'] >= self.config.max_hold_bars:
                exit_reason = 'TIME_STOP'
                exit_price = row['close']
            
            if exit_reason:
                self._close_position(pos, exit_price, exit_reason, row.get('open_time', bar_idx))
                closed.append(pos)
        
        for pos in closed:
            self.positions.remove(pos)
    
    def _close_position(self, pos, exit_price, reason, exit_time):
        """平倉"""
        if pos['direction'] == 'LONG':
            price_change = (exit_price - pos['entry_price']) / pos['entry_price']
        else:
            price_change = (pos['entry_price'] - exit_price) / pos['entry_price']
        
        pnl = pos['position_value'] * price_change
        cost = pos['position_value'] * (self.config.fee_rate + self.config.slippage) * 2
        pnl -= cost
        
        self.current_capital += pnl
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        self.trades.append({
            'entry_time': pos['entry_time'],
            'exit_time': exit_time,
            'direction': pos['direction'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'position_value': pos['position_value'],
            'pnl': pnl,
            'pnl_pct': pnl / pos['position_value'] * 100,
            'exit_reason': reason,
            'capital_after': self.current_capital
        })
    
    def _calculate_results(self, df):
        """計算結果"""
        trades_df = pd.DataFrame(self.trades)
        
        winning = trades_df[trades_df['pnl'] > 0]
        losing = trades_df[trades_df['pnl'] <= 0]
        
        win_rate = len(winning) / len(trades_df)
        profit_factor = abs(winning['pnl'].sum() / losing['pnl'].sum()) if len(losing) > 0 and losing['pnl'].sum() != 0 else float('inf')
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['peak'] = equity_df['capital'].cummax()
        equity_df['drawdown'] = (equity_df['peak'] - equity_df['capital']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].max()
        
        if 'open_time' in df.columns:
            days = (df.iloc[-1]['open_time'] - df.iloc[0]['open_time']).days
        else:
            days = len(df) / 96
        
        return {
            'status': 'success',
            'period': {
                'start': str(df.iloc[0].get('open_time', 0)),
                'end': str(df.iloc[-1].get('open_time', len(df))),
                'days': int(days)
            },
            'capital': {
                'initial': float(self.config.capital),
                'final': float(self.current_capital),
                'total_pnl': float(self.current_capital - self.config.capital),
                'total_return_pct': float((self.current_capital - self.config.capital) / self.config.capital * 100),
                'max_drawdown_pct': float(max_drawdown * 100)
            },
            'trades': {
                'total': int(len(trades_df)),
                'winning': int(len(winning)),
                'losing': int(len(losing)),
                'win_rate_pct': float(win_rate * 100),
                'avg_win': float(winning['pnl'].mean()) if len(winning) > 0 else 0,
                'avg_loss': float(losing['pnl'].mean()) if len(losing) > 0 else 0,
                'profit_factor': float(profit_factor)
            },
            'exit_reasons': trades_df['exit_reason'].value_counts().to_dict(),
            'trades_sample': trades_df.tail(10).to_dict('records')
        }
