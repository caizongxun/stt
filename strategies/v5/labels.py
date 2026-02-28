"""
V5 Label Generation
V5標籤生成 - 智能質量控制
"""
import pandas as pd
import numpy as np

class V5LabelGenerator:
    """
    智能標籤生成器
    目標: 找到高質量的交易機會
    """
    
    def __init__(self, config):
        self.config = config
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成標籤"""
        df = df.copy()
        
        print("\n[V5 Labels]")
        
        # 計算未來報酬
        df = self._calculate_future_returns(df)
        
        # 生成雙向標籤
        df = self._generate_labels(df)
        
        # 統計
        self._print_statistics(df)
        
        return df
    
    def _calculate_future_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算未來報酬"""
        forward = self.config.forward_bars
        
        # 未來最高/最低價
        df['future_high'] = df['high'].shift(-1).rolling(forward).max()
        df['future_low'] = df['low'].shift(-1).rolling(forward).min()
        
        # 未來收盤價
        df['future_close'] = df['close'].shift(-forward)
        
        # 做多潛在報酬
        df['long_return'] = (df['future_high'] - df['close']) / df['close']
        
        # 做空潛在報酬
        df['short_return'] = (df['close'] - df['future_low']) / df['close']
        
        # 做多最大回撤
        df['long_drawdown'] = (df['close'] - df['future_low']) / df['close']
        
        # 做空最大回撤
        df['short_drawdown'] = (df['future_high'] - df['close']) / df['close']
        
        return df
    
    def _generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成標籤"""
        min_return = self.config.min_return_pct
        require_no_reverse = self.config.require_no_reverse
        
        # 做多機會
        long_conditions = [
            df['long_return'] >= min_return,  # 達到目標
        ]
        
        if require_no_reverse:
            # 中間回撤不超過目標的50%
            long_conditions.append(df['long_drawdown'] <= min_return * 0.5)
        
        df['label_long'] = np.all(long_conditions, axis=0).astype(int)
        
        # 做空機會
        short_conditions = [
            df['short_return'] >= min_return,
        ]
        
        if require_no_reverse:
            short_conditions.append(df['short_drawdown'] <= min_return * 0.5)
        
        df['label_short'] = np.all(short_conditions, axis=0).astype(int)
        
        # 綜合標籤 (優先選擇更好的方向)
        df['label'] = 0
        
        # 做多機會
        df.loc[df['label_long'] == 1, 'label'] = 1
        
        # 做空機會
        df.loc[df['label_short'] == 1, 'label'] = -1
        
        # 如果兩個都是,選擇報酬更高的
        both = (df['label_long'] == 1) & (df['label_short'] == 1)
        df.loc[both & (df['long_return'] > df['short_return']), 'label'] = 1
        df.loc[both & (df['short_return'] > df['long_return']), 'label'] = -1
        
        # 轉換為二元分類 (1=有機會, 0=沒機會)
        df['label_binary'] = (df['label'] != 0).astype(int)
        
        # 保留方向資訊
        df['signal_direction'] = df['label']
        
        return df
    
    def _print_statistics(self, df: pd.DataFrame):
        """列印統計資訊"""
        valid = df['label_binary'].notna()
        total = valid.sum()
        
        if total == 0:
            print("  [WARNING] No valid labels!")
            return
        
        positive = (df.loc[valid, 'label_binary'] == 1).sum()
        positive_rate = positive / total * 100
        
        long_opp = (df.loc[valid, 'label'] == 1).sum()
        short_opp = (df.loc[valid, 'label'] == -1).sum()
        
        avg_long_return = df.loc[df['label'] == 1, 'long_return'].mean() * 100 if long_opp > 0 else 0
        avg_short_return = df.loc[df['label'] == -1, 'short_return'].mean() * 100 if short_opp > 0 else 0
        
        print(f"  Total samples: {total}")
        print(f"  Positive rate: {positive_rate:.1f}% ({positive}/{total})")
        print(f"  Long opportunities: {long_opp} (avg return: {avg_long_return:.2f}%)")
        print(f"  Short opportunities: {short_opp} (avg return: {avg_short_return:.2f}%)")
        
        if positive_rate < 15:
            print("  [WARNING] Positive rate too low! Consider lowering min_return_pct")
        elif positive_rate > 40:
            print("  [WARNING] Positive rate too high! Consider raising min_return_pct")
        else:
            print("  [OK] Label distribution looks good!")
