"""
V4 Backtester
V4回測引擎
"""
import pandas as pd
import numpy as np
from datetime import datetime

from .market_regime import MarketRegimeDetector
from .structure_detector import StructureDetector
from .signal_generator import DualModeSignalGenerator

class V4Backtester:
    """
V4回測引擎
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
        print("\n[V4 Backtesting]")
        
        # 1. 準備數據
        df = self._prepare_data(models, df, feature_names)
        print(f"  - Data prepared: {len(df)} bars")
        print(f"  - Signals found: {df['trade_signal'].sum()}")
        
        # Debug: 檢查信號分佈
        print(f"  - Long signals: {df['signal_long'].sum()}")
        print(f"  - Short signals: {df['signal_short'].sum()}")
        print(f"  - Pred > threshold: {(df['pred_proba'] >= self.config.predict_threshold).sum()}")
        
        # 2. 模擬交易
        self._simulate_trading(df)
        print(f"  - Total trades: {len(self.trades)}")
        
        # 3. 計算結果
        if len(self.trades) == 0:
            return {
                'status': 'no_trades', 
                'message': '無交易信號',
                'debug': {
                    'total_bars': len(df),
                    'signals_generated': int(df['trade_signal'].sum()),
                    'long_signals': int(df['signal_long'].sum()),
                    'short_signals': int(df['signal_short'].sum()),
                    'high_confidence': int((df['pred_proba'] >= self.config.predict_threshold).sum()),
                    'sample_probas': df['pred_proba'].describe().to_dict()
                }
            }
        
        results = self._calculate_results(df)
        
        return results
    
    def _prepare_data(self, models, df, feature_names) -> pd.DataFrame:
        """準備回測數據"""
        df = df.copy()
        
        # 市場狀態識別
        regime_detector = MarketRegimeDetector(self.config)
        df = regime_detector.detect(df)
        
        # 結構識別
        structure_detector = StructureDetector(self.config)
        df = structure_detector.detect(df)
        
        # 信號生成
        signal_gen = DualModeSignalGenerator(self.config)
        df = signal_gen.generate(df)
        
        # 模型預測
        X = df[feature_names].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())
        
        # 集成預測
        if len(models) > 1:
            probas = [m.predict_proba(X)[:, 1] for m in models]
            df['pred_proba'] = np.mean(probas, axis=0)
        else:
            df['pred_proba'] = models[0].predict_proba(X)[:, 1]
        
        # 交易信號 - 修正邏輯
        # 根據模型預測 + 原始信號方向
        high_confidence = df['pred_proba'] >= self.config.predict_threshold
        
        # 重新生成signal_long/short
        df['signal_long'] = 0
        df['signal_short'] = 0
        
        # 盤整模式
        ranging = df['regime'] == 'RANGING'
        df.loc[ranging & high_confidence & (df['near_support'] == 1) & (df['rsi'] < 40), 'signal_long'] = 1
        df.loc[ranging & high_confidence & (df['near_resistance'] == 1) & (df['rsi'] > 60), 'signal_short'] = 1
        
        # 趨勢模式
        trending = df['regime'] == 'TRENDING'
        df.loc[trending & high_confidence & (df['breakout_up'] == 1) & (df['volume_ratio'] > 1.2), 'signal_long'] = 1
        df.loc[trending & high_confidence & (df['breakout_down'] == 1) & (df['volume_ratio'] > 1.2), 'signal_short'] = 1
        
        # 總信號
        df['trade_signal'] = ((df['signal_long'] == 1) | (df['signal_short'] == 1)).astype(int)
        
        return df
    
    def _simulate_trading(self, df: pd.DataFrame):
        """模擬交易執行"""
        for i in range(len(df)):
            row = df.iloc[i]
            
            # 重置每日計數
            if i > 0:
                current_date = pd.to_datetime(row.get('open_time', i)).date() if 'open_time' in df.columns else i // 96
                prev_date = pd.to_datetime(df.iloc[i-1].get('open_time', i-1)).date() if 'open_time' in df.columns else (i-1) // 96
                if current_date != prev_date:
                    self.daily_trades = 0
            
            # 管理持仓
            self._manage_positions(row, i)
            
            # 檢查開仓
            if len(self.positions) < self.config.max_positions and \
               self.daily_trades < self.config.max_trades_per_day and \
               i - self.last_trade_bar >= 4:  # 至少4根K棒間隔
                
                if row['trade_signal'] == 1:
                    # 根據信號方向開仓
                    if row['signal_long'] == 1:
                        self._open_position(row, i, 'LONG')
                    elif row['signal_short'] == 1:
                        self._open_position(row, i, 'SHORT')
            
            # 記錄權益
            self.equity_curve.append({
                'bar': i,
                'time': row.get('open_time', i),
                'capital': self.current_capital,
                'positions': len(self.positions)
            })
    
    def _open_position(self, row, bar_idx: int, direction: str):
        """開仓"""
        entry_price = row['close']
        atr = row['atr'] if not pd.isna(row['atr']) else entry_price * 0.02
        
        # 根據市場狀態設置止盈目標
        if row['regime'] == 'RANGING':
            tp_multiplier = self.config.atr_tp_range
        else:
            tp_multiplier = self.config.atr_tp_breakout
        
        # 計算止損止盈
        if direction == 'LONG':
            stop_loss = entry_price - atr * self.config.atr_sl_multiplier
            take_profit = entry_price + atr * tp_multiplier
        else:
            stop_loss = entry_price + atr * self.config.atr_sl_multiplier
            take_profit = entry_price - atr * tp_multiplier
        
        # 仓位大小
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
            'regime': row['regime'],
            'atr': atr
        }
        
        self.positions.append(position)
        self.daily_trades += 1
        self.last_trade_bar = bar_idx
    
    def _manage_positions(self, row, bar_idx: int):
        """管理持仓"""
        closed = []
        
        for pos in self.positions:
            exit_reason = None
            exit_price = None
            
            # 止損
            if pos['direction'] == 'LONG' and row['low'] <= pos['stop_loss']:
                exit_reason = 'STOP_LOSS'
                exit_price = pos['stop_loss']
            elif pos['direction'] == 'SHORT' and row['high'] >= pos['stop_loss']:
                exit_reason = 'STOP_LOSS'
                exit_price = pos['stop_loss']
            
            # 止盈
            elif pos['direction'] == 'LONG' and row['high'] >= pos['take_profit']:
                exit_reason = 'TAKE_PROFIT'
                exit_price = pos['take_profit']
            elif pos['direction'] == 'SHORT' and row['low'] <= pos['take_profit']:
                exit_reason = 'TAKE_PROFIT'
                exit_price = pos['take_profit']
            
            # 時間止損 (20根K棒 = 5小時)
            elif bar_idx - pos['entry_bar'] >= 20:
                exit_reason = 'TIME_STOP'
                exit_price = row['close']
            
            # 平仓
            if exit_reason:
                self._close_position(pos, exit_price, exit_reason, row.get('open_time', bar_idx))
                closed.append(pos)
        
        for pos in closed:
            self.positions.remove(pos)
    
    def _close_position(self, pos, exit_price, reason, exit_time):
        """平仓"""
        # 計算盈虧
        if pos['direction'] == 'LONG':
            price_change = (exit_price - pos['entry_price']) / pos['entry_price']
        else:
            price_change = (pos['entry_price'] - exit_price) / pos['entry_price']
        
        # 含槓桶
        pnl = pos['position_value'] * price_change
        
        # 手續費
        cost = pos['position_value'] * (self.config.fee_rate + self.config.slippage) * 2
        pnl -= cost
        
        # 更新資金
        self.current_capital += pnl
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # 記錄
        self.trades.append({
            'entry_time': pos['entry_time'],
            'exit_time': exit_time,
            'direction': pos['direction'],
            'regime': pos['regime'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'position_value': pos['position_value'],
            'pnl': pnl,
            'pnl_pct': pnl / pos['position_value'] * 100,
            'exit_reason': reason,
            'capital_after': self.current_capital
        })
    
    def _calculate_results(self, df) -> dict:
        """計算回測結果"""
        trades_df = pd.DataFrame(self.trades)
        
        # 基本統計
        total_trades = len(trades_df)
        winning = trades_df[trades_df['pnl'] > 0]
        losing = trades_df[trades_df['pnl'] <= 0]
        
        win_rate = len(winning) / total_trades if total_trades > 0 else 0
        
        # 盈虧
        total_pnl = trades_df['pnl'].sum()
        avg_win = winning['pnl'].mean() if len(winning) > 0 else 0
        avg_loss = losing['pnl'].mean() if len(losing) > 0 else 0
        
        profit_factor = abs(winning['pnl'].sum() / losing['pnl'].sum()) if len(losing) > 0 and losing['pnl'].sum() != 0 else float('inf')
        
        # 報酬率
        initial = self.config.capital
        final = self.current_capital
        total_return = (final - initial) / initial
        
        # 最大回撤
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['peak'] = equity_df['capital'].cummax()
        equity_df['drawdown'] = (equity_df['peak'] - equity_df['capital']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].max()
        
        # 分狀態統計
        ranging_trades = trades_df[trades_df['regime'] == 'RANGING']
        trending_trades = trades_df[trades_df['regime'] == 'TRENDING']
        
        ranging_win_rate = (ranging_trades['pnl'] > 0).sum() / len(ranging_trades) if len(ranging_trades) > 0 else 0
        trending_win_rate = (trending_trades['pnl'] > 0).sum() / len(trending_trades) if len(trending_trades) > 0 else 0
        
        # 時間
        if 'open_time' in df.columns:
            days = (df.iloc[-1]['open_time'] - df.iloc[0]['open_time']).days
        else:
            days = len(df) / 96  # 15m = 96根/天
        
        return {
            'status': 'success',
            'period': {
                'start': str(df.iloc[0].get('open_time', 0)),
                'end': str(df.iloc[-1].get('open_time', len(df))),
                'days': int(days)
            },
            'capital': {
                'initial': float(initial),
                'final': float(final),
                'total_pnl': float(total_pnl),
                'total_return_pct': float(total_return * 100),
                'max_drawdown_pct': float(max_drawdown * 100)
            },
            'trades': {
                'total': int(total_trades),
                'winning': int(len(winning)),
                'losing': int(len(losing)),
                'win_rate_pct': float(win_rate * 100),
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss),
                'profit_factor': float(profit_factor)
            },
            'regime_performance': {
                'ranging_trades': int(len(ranging_trades)),
                'ranging_win_rate': float(ranging_win_rate * 100),
                'ranging_pnl': float(ranging_trades['pnl'].sum()) if len(ranging_trades) > 0 else 0,
                'trending_trades': int(len(trending_trades)),
                'trending_win_rate': float(trending_win_rate * 100),
                'trending_pnl': float(trending_trades['pnl'].sum()) if len(trending_trades) > 0 else 0
            },
            'exit_reasons': trades_df['exit_reason'].value_counts().to_dict(),
            'trades_sample': trades_df.tail(10).to_dict('records')
        }
