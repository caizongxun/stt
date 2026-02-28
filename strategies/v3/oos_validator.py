"""
V3 OOS Validator
V3 OOS验证器 - Walk-Forward分析
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class OOSValidator:
    """
    Out-of-Sample验证器
    Walk-Forward滚动窗口验证
    """
    
    def __init__(self, config):
        self.config = config
    
    def split_data(self, df: pd.DataFrame) -> dict:
        """
        分割数据集
        Train: 6个月
        Val: 1个月
        OOS: 1个月 (完全未见)
        """
        total_bars = len(df)
        bars_per_month = 30 * 24 * 4  # 30天 * 24小时 * 4个15分钟
        
        train_bars = self.config.train_months * bars_per_month
        val_bars = self.config.val_months * bars_per_month
        oos_bars = self.config.oos_months * bars_per_month
        
        # 确保有足够数据
        required_bars = train_bars + val_bars + oos_bars
        if total_bars < required_bars:
            # 调整到可用数据
            train_bars = int(total_bars * 0.75)
            val_bars = int(total_bars * 0.125)
            oos_bars = total_bars - train_bars - val_bars
        
        # 分割
        train_end = train_bars
        val_end = train_end + val_bars
        
        return {
            'train': df.iloc[:train_end].copy(),
            'val': df.iloc[train_end:val_end].copy(),
            'oos': df.iloc[val_end:val_end+oos_bars].copy(),
            'split_info': {
                'total_bars': total_bars,
                'train_bars': train_end,
                'val_bars': val_bars,
                'oos_bars': oos_bars,
                'train_pct': train_end / total_bars * 100,
                'val_pct': val_bars / total_bars * 100,
                'oos_pct': oos_bars / total_bars * 100
            }
        }
    
    def walk_forward_split(self, df: pd.DataFrame, n_splits: int = 3) -> list:
        """
        Walk-Forward分割
        滚动窗口,每次前进1个月
        """
        splits = []
        bars_per_month = 30 * 24 * 4
        
        train_bars = self.config.train_months * bars_per_month
        val_bars = self.config.val_months * bars_per_month
        oos_bars = self.config.oos_months * bars_per_month
        
        for i in range(n_splits):
            offset = i * bars_per_month
            
            if offset + train_bars + val_bars + oos_bars > len(df):
                break
            
            train_start = offset
            train_end = offset + train_bars
            val_end = train_end + val_bars
            oos_end = val_end + oos_bars
            
            splits.append({
                'split_id': i,
                'train': df.iloc[train_start:train_end].copy(),
                'val': df.iloc[train_end:val_end].copy(),
                'oos': df.iloc[val_end:oos_end].copy()
            })
        
        return splits
    
    def validate_no_leakage(self, train_df: pd.DataFrame, val_df: pd.DataFrame, oos_df: pd.DataFrame) -> dict:
        """
        验证数据没有泄漏
        """
        issues = []
        
        # 检查时间顺序
        if 'open_time' in train_df.columns:
            train_max_time = train_df['open_time'].max()
            val_min_time = val_df['open_time'].min()
            oos_min_time = oos_df['open_time'].min()
            
            if train_max_time >= val_min_time:
                issues.append('Train and Val overlap in time')
            if val_df['open_time'].max() >= oos_min_time:
                issues.append('Val and OOS overlap in time')
        
        # 检查索引不重复
        train_idx = set(train_df.index)
        val_idx = set(val_df.index)
        oos_idx = set(oos_df.index)
        
        if train_idx & val_idx:
            issues.append(f'Train and Val share {len(train_idx & val_idx)} indices')
        if val_idx & oos_idx:
            issues.append(f'Val and OOS share {len(val_idx & oos_idx)} indices')
        if train_idx & oos_idx:
            issues.append(f'Train and OOS share {len(train_idx & oos_idx)} indices')
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues
        }
