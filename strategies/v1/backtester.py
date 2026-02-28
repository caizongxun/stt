"""
V1 Backtester
V1回測模組
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class Backtester:
    """
    V1回測器
    """
    
    def __init__(self, config):
        self.config = config
        self.trades = []
        self.equity_curve = []
    
    def run(self, model, df: pd.DataFrame, feature_names: list) -> dict:
        """
        執行回測
        
        Args:
            model: 訓練好的模型
            df: K線數據
            feature_names: 特徵名稱列表
        
        Returns:
            dict: 詳細回測結果
        """
        # 1. 準備數據
        df = self._prepare_data(df, feature_names)
        
        # 2. 生成信號
        df['signal'] = model.predict(df[feature_names].values)
        df['signal_proba'] = model.predict_proba(df[feature_names].values).max(axis=1)
        
        # 3. 模擬交易
        self._simulate_trading(df)
        
        # 4. 計算績效
        metrics = self._calculate_metrics()
        
        # 5. 組裝結果
        results = {
            "backtest_info": {
                "symbol": self.config.symbol,
                "timeframe": self.config.timeframe,
                "start_date": str(df['open_time'].iloc[0]),
                "end_date": str(df['open_time'].iloc[-1]),
                "total_bars": len(df),
                "backtest_days": self.config.backtest_days,
                "backtest_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            "trading_params": {
                "initial_capital": self.config.capital,
                "leverage": self.config.leverage,
                "fee_rate": self.config.fee_rate
            },
            "performance_metrics": metrics,
            "trades_summary": self._get_trades_summary(),
            "equity_curve": self._get_equity_curve_data(),
            "optimization_suggestions": self._generate_backtest_suggestions(metrics)
        }
        
        return results
    
    def _prepare_data(self, df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
        """準備回測數據"""
        df = df.copy()
        
        # 特徵工程 (與訓練時相同)
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        for period in self.config.lookback_periods:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'std_{period}'] = df['close'].rolling(period).std()
            df[f'momentum_{period}'] = df['close'].pct_change(period)
        
        df = df.dropna()
        
        # 只取最後 N 天
        if self.config.backtest_days > 0:
            bars_per_day = {
                '1m': 1440,
                '15m': 96,
                '1h': 24,
                '1d': 1
            }
            n_bars = self.config.backtest_days * bars_per_day.get(self.config.timeframe, 96)
            df = df.tail(n_bars)
        
        return df
    
    def _simulate_trading(self, df: pd.DataFrame):
        """模擬交易過程"""
        capital = self.config.capital
        position = 0  # 0: 空倉, 1: 多倉, -1: 空倉
        entry_price = 0
        entry_time = None
        
        self.trades = []
        self.equity_curve = []
        
        for idx, row in df.iterrows():
            current_price = row['close']
            current_time = row['open_time']
            signal = row['signal']
            
            # 記錄權益
            current_equity = capital
            if position != 0:
                pnl_pct = (current_price / entry_price - 1) * position
                current_equity = capital * (1 + pnl_pct * self.config.leverage)
            
            self.equity_curve.append({
                'timestamp': current_time,
                'equity': current_equity,
                'position': position
            })
            
            # 交易逻輯
            if position == 0:
                # 無位置,根據信號開倉
                if signal == 1:  # 做多
                    position = 1
                    entry_price = current_price
                    entry_time = current_time
                elif signal == 2:  # 做空
                    position = -1
                    entry_price = current_price
                    entry_time = current_time
            else:
                # 有位置,檢查是否平倉
                should_close = False
                
                # 信號反轉
                if (position == 1 and signal == 2) or (position == -1 and signal == 1):
                    should_close = True
                
                # 信號變為持有
                if signal == 0:
                    should_close = True
                
                if should_close:
                    # 平倉
                    pnl_pct = (current_price / entry_price - 1) * position
                    pnl_with_leverage = pnl_pct * self.config.leverage
                    
                    # 扣除手續費 (開倉+平倉)
                    fee = self.config.fee_rate * 2 * self.config.leverage
                    net_pnl_pct = pnl_with_leverage - fee
                    
                    pnl = capital * net_pnl_pct
                    capital += pnl
                    
                    # 記錄交易
                    self.trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'direction': 'LONG' if position == 1 else 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct,
                        'pnl_with_leverage': pnl_with_leverage,
                        'fee': fee,
                        'net_pnl_pct': net_pnl_pct,
                        'pnl': pnl,
                        'capital_after': capital
                    })
                    
                    # 重置位置
                    position = 0
                    entry_price = 0
                    entry_time = None
        
        # 最後如果還有持倉,強制平倉
        if position != 0:
            current_price = df.iloc[-1]['close']
            current_time = df.iloc[-1]['open_time']
            
            pnl_pct = (current_price / entry_price - 1) * position
            pnl_with_leverage = pnl_pct * self.config.leverage
            fee = self.config.fee_rate * 2 * self.config.leverage
            net_pnl_pct = pnl_with_leverage - fee
            pnl = capital * net_pnl_pct
            capital += pnl
            
            self.trades.append({
                'entry_time': entry_time,
                'exit_time': current_time,
                'direction': 'LONG' if position == 1 else 'SHORT',
                'entry_price': entry_price,
                'exit_price': current_price,
                'pnl_pct': pnl_pct,
                'pnl_with_leverage': pnl_with_leverage,
                'fee': fee,
                'net_pnl_pct': net_pnl_pct,
                'pnl': pnl,
                'capital_after': capital
            })
    
    def _calculate_metrics(self) -> dict:
        """計算績效指標"""
        if not self.trades:
            return {
                "total_return": 0.0,
                "total_return_pct": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "max_win": 0.0,
                "max_loss": 0.0
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        # 基礎指標
        final_capital = trades_df['capital_after'].iloc[-1]
        total_return = final_capital - self.config.capital
        total_return_pct = (final_capital / self.config.capital - 1) * 100
        
        # 勝率
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        win_rate = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        
        # 盈虧比
        total_profit = wins['pnl'].sum() if len(wins) > 0 else 0
        total_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        # Sharpe Ratio
        returns = trades_df['net_pnl_pct'].values
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # 最大回撤
        equity_curve = pd.DataFrame(self.equity_curve)
        equity_curve['cummax'] = equity_curve['equity'].cummax()
        equity_curve['drawdown'] = (equity_curve['equity'] - equity_curve['cummax']) / equity_curve['cummax']
        max_drawdown = abs(equity_curve['drawdown'].min()) * 100
        
        # 平均盈虧
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        max_win = wins['pnl'].max() if len(wins) > 0 else 0
        max_loss = losses['pnl'].min() if len(losses) > 0 else 0
        
        return {
            "total_return": float(total_return),
            "total_return_pct": float(total_return_pct),
            "total_trades": len(trades_df),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "max_win": float(max_win),
            "max_loss": float(max_loss),
            "final_capital": float(final_capital)
        }
    
    def _get_trades_summary(self) -> dict:
        """獲取交易摘要"""
        if not self.trades:
            return {"recent_trades": [], "total_trades": 0}
        
        return {
            "recent_trades": self.trades[-10:],  # 最近10筆交易
            "total_trades": len(self.trades)
        }
    
    def _get_equity_curve_data(self) -> list:
        """獲取權益曲線數據"""
        # 每100個點取一個,減少數據量
        step = max(1, len(self.equity_curve) // 100)
        return self.equity_curve[::step]
    
    def _generate_backtest_suggestions(self, metrics: dict) -> list:
        """生成回測優化建議"""
        suggestions = []
        
        # 1. 總報酬
        if metrics['total_return_pct'] < 0:
            suggestions.append({
                "issue": "negative_return",
                "description": f"總報酬為負({metrics['total_return_pct']:.2f}%)",
                "recommendation": "建議: 重新訓練模型或調整特徵工程"
            })
        
        # 2. 勝率
        if metrics['win_rate'] < 40:
            suggestions.append({
                "issue": "low_win_rate",
                "description": f"勝率過低({metrics['win_rate']:.1f}%)",
                "recommendation": "建議: 優化進場條件或增加信號篩選"
            })
        
        # 3. 盈虧比
        if metrics['profit_factor'] < 1.0:
            suggestions.append({
                "issue": "low_profit_factor",
                "description": f"盈虧比<1({metrics['profit_factor']:.2f})",
                "recommendation": "建議: 調整止盈止損比例或優化出場逻輯"
            })
        
        # 4. 最大回撤
        if metrics['max_drawdown'] > 30:
            suggestions.append({
                "issue": "high_drawdown",
                "description": f"最大回撤過大({metrics['max_drawdown']:.1f}%)",
                "recommendation": "建議: 降低槓桿或增加風控機制"
            })
        
        # 5. Sharpe Ratio
        if metrics['sharpe_ratio'] < 1.0:
            suggestions.append({
                "issue": "low_sharpe",
                "description": f"Sharpe比率過低({metrics['sharpe_ratio']:.2f})",
                "recommendation": "建議: 優化風險調整報酬比"
            })
        
        # 6. 交易次數
        if metrics['total_trades'] < 10:
            suggestions.append({
                "issue": "few_trades",
                "description": f"交易次數過少({metrics['total_trades']})",
                "recommendation": "建議: 放寬交易條件或增加回測期間"
            })
        
        if not suggestions:
            suggestions.append({
                "issue": "none",
                "description": "回測結果良好",
                "recommendation": "可以考慮實盤交易前做更多測試"
            })
        
        return suggestions
