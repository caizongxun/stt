"""
V3 OOS Validator
V3 OOS驗證器 - Walk-Forward分析
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class OOSValidator:
    """
    Out-of-Sample驗證器
    Walk-Forward滾動窗口驗證
    """
    
    def __init__(self, config):
        self.config = config
    
    def split_data(self, df: pd.DataFrame) -> dict:
        """
        分割數據集 - 修正版
        Train: 75%
        Val: 12.5%
        OOS: 12.5%
        """
        total_bars = len(df)
        
        # 簡單比例分割
        train_bars = int(total_bars * 0.75)
        val_bars = int(total_bars * 0.125)
        oos_bars = total_bars - train_bars - val_bars
        
        # 分割
        train_end = train_bars
        val_end = train_end + val_bars
        
        return {
            'train': df.iloc[:train_end].copy(),
            'val': df.iloc[train_end:val_end].copy(),
            'oos': df.iloc[val_end:].copy(),
            'split_info': {
                'total_bars': total_bars,
                'train_bars': train_bars,
                'val_bars': val_bars,
                'oos_bars': oos_bars,
                'train_pct': train_bars / total_bars * 100,
                'val_pct': val_bars / total_bars * 100,
                'oos_pct': oos_bars / total_bars * 100
            }
        }
    
    def walk_forward_split(self, df: pd.DataFrame, n_splits: int = 3) -> list:
        """
        Walk-Forward分割
        滾動窗口,每次前進一段
        """
        splits = []
        total_bars = len(df)
        
        segment_size = total_bars // (n_splits + 2)
        
        for i in range(n_splits):
            train_start = i * segment_size
            train_end = train_start + segment_size * 3
            val_end = train_end + segment_size
            oos_end = val_end + segment_size
            
            if oos_end > total_bars:
                break
            
            splits.append({
                'split_id': i,
                'train': df.iloc[train_start:train_end].copy(),
                'val': df.iloc[train_end:val_end].copy(),
                'oos': df.iloc[val_end:oos_end].copy()
            })
        
        return splits
    
    def validate_no_leakage(self, train_df: pd.DataFrame, val_df: pd.DataFrame, oos_df: pd.DataFrame) -> dict:
        """
        驗證數據沒有洩漏
        """
        issues = []
        
        # 檢查時間順序
        if 'open_time' in train_df.columns:
            train_max_time = train_df['open_time'].max()
            val_min_time = val_df['open_time'].min()
            oos_min_time = oos_df['open_time'].min()
            
            if train_max_time >= val_min_time:
                issues.append('Train and Val overlap in time')
            if val_df['open_time'].max() >= oos_min_time:
                issues.append('Val and OOS overlap in time')
        
        # 檢查索引不重複
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
