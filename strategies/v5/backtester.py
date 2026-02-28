"""
V5 Backtester - Dual Model
"""
import pandas as pd
import numpy as np
from .features import V5FeatureEngine

class V5Backtester:
    def __init__(self, config):
        self.config = config
        self.trades = []
        self.positions = []
        self.equity_curve = []
        self.current_capital = config.capital
        self.peak_capital = config.capital
        self.daily_trades = 0
        self.last_trade_bar = -999
    
    def run(self, long_models, short_models, df, feature_names):
        print("\n[V5 Backtesting]")
        df = self._prepare_data(long_models, short_models, df, feature_names)
        print(f"  Bars: {len(df)}")
        print(f"  Long: {df['long_signal'].sum()}")
        print(f"  Short: {df['short_signal'].sum()}")
        
        self._simulate_trading(df)
        print(f"  Trades: {len(self.trades)}")
        
        if len(self.trades) == 0:
            return {'status': 'no_trades'}
        return self._calculate_results(df)
    
    def _prepare_data(self, long_models, short_models, df, feature_names):
        df = df.copy()
        feature_engine = V5FeatureEngine(self.config)
        df = feature_engine.generate(df)
        
        X = df[feature_names].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())
        
        long_probas = [m.predict_proba(X)[:, 1] for m in long_models]
        df['long_proba'] = np.mean(long_probas, axis=0)
        df['long_signal'] = (df['long_proba'] >= self.config.long_threshold).astype(int)
        
        short_probas = [m.predict_proba(X)[:, 1] for m in short_models]
        df['short_proba'] = np.mean(short_probas, axis=0)
        df['short_signal'] = (df['short_proba'] >= self.config.short_threshold).astype(int)
        
        return df
    
    def _simulate_trading(self, df):
        for i in range(len(df)):
            row = df.iloc[i]
            
            if i > 0:
                current_date = pd.to_datetime(row.get('open_time', i)).date() if 'open_time' in df.columns else i // 96
                prev_date = pd.to_datetime(df.iloc[i-1].get('open_time', i-1)).date() if 'open_time' in df.columns else (i-1) // 96
                if current_date != prev_date:
                    self.daily_trades = 0
            
            self._manage_positions(row, i)
            
            can_open = (len(self.positions) < self.config.max_positions and
                       self.daily_trades < self.config.max_trades_per_day and
                       i - self.last_trade_bar >= self.config.min_bars_between)
            
            if can_open:
                if row['long_signal'] == 1:
                    self._open_position(row, i, 'LONG', row['long_proba'])
                elif row['short_signal'] == 1:
                    self._open_position(row, i, 'SHORT', row['short_proba'])
            
            self.equity_curve.append({'bar': i, 'time': row.get('open_time', i), 'capital': self.current_capital, 'positions': len(self.positions)})
    
    def _open_position(self, row, bar_idx, direction, confidence):
        entry_price = row['close']
        atr = row.get('atr', entry_price * 0.02)
        if pd.isna(atr) or atr == 0:
            atr = entry_price * 0.02
        
        if direction == 'LONG':
            stop_loss = entry_price - atr * self.config.atr_sl_multiplier
            take_profit = entry_price + atr * self.config.atr_tp_multiplier
        else:
            stop_loss = entry_price + atr * self.config.atr_sl_multiplier
            take_profit = entry_price - atr * self.config.atr_tp_multiplier
        
        available = self.current_capital if self.config.use_compound else self.config.capital
        position_value = available * self.config.position_pct * self.config.leverage
        
        self.positions.append({'direction': direction, 'entry_bar': bar_idx, 'entry_price': entry_price,
                               'entry_time': row.get('open_time', bar_idx), 'position_value': position_value,
                               'stop_loss': stop_loss, 'take_profit': take_profit, 'trailing_stop': None,
                               'atr': atr, 'confidence': confidence})
        self.daily_trades += 1
        self.last_trade_bar = bar_idx
    
    def _manage_positions(self, row, bar_idx):
        closed = []
        for pos in self.positions:
            current_profit = (row['close'] - pos['entry_price']) / pos['entry_price'] if pos['direction'] == 'LONG' else (pos['entry_price'] - row['close']) / pos['entry_price']
            
            if self.config.use_trailing_stop and current_profit > self.config.trailing_activation:
                if pos['trailing_stop'] is None:
                    pos['trailing_stop'] = row['close'] - row['close'] * self.config.trailing_distance if pos['direction'] == 'LONG' else row['close'] + row['close'] * self.config.trailing_distance
                else:
                    if pos['direction'] == 'LONG':
                        new_trailing = row['close'] - row['close'] * self.config.trailing_distance
                        if new_trailing > pos['trailing_stop']:
                            pos['trailing_stop'] = new_trailing
                    else:
                        new_trailing = row['close'] + row['close'] * self.config.trailing_distance
                        if new_trailing < pos['trailing_stop']:
                            pos['trailing_stop'] = new_trailing
            
            exit_reason = None
            exit_price = None
            
            if pos['trailing_stop'] is not None:
                if (pos['direction'] == 'LONG' and row['low'] <= pos['trailing_stop']) or (pos['direction'] == 'SHORT' and row['high'] >= pos['trailing_stop']):
                    exit_reason = 'TRAILING_STOP'
                    exit_price = pos['trailing_stop']
            
            if not exit_reason:
                if (pos['direction'] == 'LONG' and row['low'] <= pos['stop_loss']) or (pos['direction'] == 'SHORT' and row['high'] >= pos['stop_loss']):
                    exit_reason = 'STOP_LOSS'
                    exit_price = pos['stop_loss']
            
            if not exit_reason:
                if (pos['direction'] == 'LONG' and row['high'] >= pos['take_profit']) or (pos['direction'] == 'SHORT' and row['low'] <= pos['take_profit']):
                    exit_reason = 'TAKE_PROFIT'
                    exit_price = pos['take_profit']
            
            if not exit_reason and bar_idx - pos['entry_bar'] >= self.config.max_hold_bars:
                exit_reason = 'TIME_STOP'
                exit_price = row['close']
            
            if exit_reason:
                self._close_position(pos, exit_price, exit_reason, row.get('open_time', bar_idx))
                closed.append(pos)
        
        for pos in closed:
            self.positions.remove(pos)
    
    def _close_position(self, pos, exit_price, reason, exit_time):
        price_change = (exit_price - pos['entry_price']) / pos['entry_price'] if pos['direction'] == 'LONG' else (pos['entry_price'] - exit_price) / pos['entry_price']
        pnl = pos['position_value'] * price_change
        cost = pos['position_value'] * (self.config.fee_rate + self.config.slippage) * 2
        pnl -= cost
        
        self.current_capital += pnl
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        self.trades.append({'entry_time': pos['entry_time'], 'exit_time': exit_time, 'direction': pos['direction'],
                           'entry_price': pos['entry_price'], 'exit_price': exit_price, 'position_value': pos['position_value'],
                           'pnl': pnl, 'pnl_pct': pnl / pos['position_value'] * 100, 'exit_reason': reason,
                           'confidence': pos['confidence'], 'capital_after': self.current_capital})
    
    def _calculate_results(self, df):
        trades_df = pd.DataFrame(self.trades)
        winning = trades_df[trades_df['pnl'] > 0]
        losing = trades_df[trades_df['pnl'] <= 0]
        win_rate = len(winning) / len(trades_df)
        profit_factor = abs(winning['pnl'].sum() / losing['pnl'].sum()) if len(losing) > 0 and losing['pnl'].sum() != 0 else float('inf')
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['peak'] = equity_df['capital'].cummax()
        equity_df['drawdown'] = (equity_df['peak'] - equity_df['capital']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].max()
        
        days = (df.iloc[-1]['open_time'] - df.iloc[0]['open_time']).days if 'open_time' in df.columns else len(df) / 96
        total_return = (self.current_capital - self.config.capital) / self.config.capital * 100
        monthly_return = total_return * 30 / days if days > 0 else 0
        
        long_trades = trades_df[trades_df['direction'] == 'LONG']
        short_trades = trades_df[trades_df['direction'] == 'SHORT']
        
        return {
            'status': 'success',
            'period': {'start': str(df.iloc[0].get('open_time', 0)), 'end': str(df.iloc[-1].get('open_time', len(df))), 'days': int(days)},
            'capital': {'initial': float(self.config.capital), 'final': float(self.current_capital),
                       'total_pnl': float(self.current_capital - self.config.capital),
                       'total_return_pct': float(total_return), 'monthly_return_pct': float(monthly_return),
                       'max_drawdown_pct': float(max_drawdown * 100)},
            'trades': {'total': int(len(trades_df)), 'winning': int(len(winning)), 'losing': int(len(losing)),
                      'win_rate_pct': float(win_rate * 100), 'avg_win': float(winning['pnl'].mean()) if len(winning) > 0 else 0,
                      'avg_loss': float(losing['pnl'].mean()) if len(losing) > 0 else 0, 'profit_factor': float(profit_factor),
                      'long_trades': int(len(long_trades)), 'short_trades': int(len(short_trades)),
                      'long_win_rate': float((long_trades['pnl'] > 0).sum() / len(long_trades) * 100) if len(long_trades) > 0 else 0,
                      'short_win_rate': float((short_trades['pnl'] > 0).sum() / len(short_trades) * 100) if len(short_trades) > 0 else 0},
            'exit_reasons': trades_df['exit_reason'].value_counts().to_dict(),
            'trades_sample': trades_df.tail(15).to_dict('records')}
