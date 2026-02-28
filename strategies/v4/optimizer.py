"""
V4 Parameter Optimizer
V4參數優化器
"""
import pandas as pd
import numpy as np
from itertools import product
from .backtester import V4Backtester
from .config import V4Config

class ParameterOptimizer:
    """參數網格搜索優化器"""
    
    def __init__(self, base_config: V4Config):
        self.base_config = base_config
        self.results = []
    
    def optimize(self, models, df: pd.DataFrame, feature_names: list, 
                 param_grid: dict = None) -> dict:
        """
        網格搜索最佳參數
        """
        if param_grid is None:
            param_grid = self._get_default_grid()
        
        print("\n[V4 Parameter Optimization]")
        print(f"Testing {self._count_combinations(param_grid)} combinations...\n")
        
        # 生成所有參數組合
        param_combinations = self._generate_combinations(param_grid)
        
        best_result = None
        best_score = -float('inf')
        
        for i, params in enumerate(param_combinations, 1):
            # 更新配置
            config = V4Config(**self.base_config.to_dict())
            for key, value in params.items():
                setattr(config, key, value)
            
            # 回測
            backtester = V4Backtester(config)
            result = backtester.run(models, df.copy(), feature_names)
            
            if result['status'] == 'success':
                # 計算評分
                score = self._calculate_score(result)
                
                result['params'] = params
                result['score'] = score
                self.results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_result = result
                
                # 進度顯示
                if i % 5 == 0 or score > best_score:
                    print(f"  [{i}/{len(param_combinations)}] Score: {score:.2f} | "
                          f"Return: {result['capital']['total_return_pct']:.1f}% | "
                          f"WinRate: {result['trades']['win_rate_pct']:.1f}% | "
                          f"PF: {result['trades']['profit_factor']:.2f}")
        
        print(f"\n[Optimization Complete]")
        print(f"Best Score: {best_score:.2f}")
        print(f"Best Params: {best_result['params']}")
        
        return {
            'best_result': best_result,
            'all_results': sorted(self.results, key=lambda x: x['score'], reverse=True)[:10],
            'best_params': best_result['params']
        }
    
    def _get_default_grid(self) -> dict:
        """預設參數網格"""
        return {
            'predict_threshold': [0.45, 0.50, 0.55],
            'atr_sl_multiplier': [1.0, 1.2, 1.5],
            'atr_tp_range': [2.0, 2.5, 3.0],
            'atr_tp_breakout': [3.0, 4.0, 5.0],
            'leverage': [2, 3, 4],
            'position_pct': [0.25, 0.30, 0.35]
        }
    
    def _generate_combinations(self, param_grid: dict) -> list:
        """生成所有參數組合"""
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = []
        
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _count_combinations(self, param_grid: dict) -> int:
        """計算組合數"""
        count = 1
        for values in param_grid.values():
            count *= len(values)
        return count
    
    def _calculate_score(self, result: dict) -> float:
        """
        計算綜合評分
        考慮: 報酬率, 勝率, 利潤因子, 回撤
        """
        ret = result['capital']['total_return_pct']
        win_rate = result['trades']['win_rate_pct']
        pf = result['trades']['profit_factor']
        dd = result['capital']['max_drawdown_pct']
        trades = result['trades']['total']
        
        # 基本篩選
        if trades < 10:
            return -1000  # 交易太少
        if win_rate < 45:
            return -1000  # 勝率太低
        if pf < 1.0:
            return -1000  # 虧錢
        if dd > 25:
            return -1000  # 回撤太大
        
        # 綜合評分 (權重)
        score = (
            ret * 0.4 +          # 40% 報酬率
            win_rate * 0.2 +     # 20% 勝率
            pf * 10 * 0.2 +      # 20% 利潤因子
            (100 - dd) * 0.2     # 20% 回撤控制
        )
        
        return score
    
    def get_optimization_report(self) -> pd.DataFrame:
        """獲取優化報告"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for r in self.results:
            if r['status'] == 'success':
                row = {
                    'score': r['score'],
                    'return_pct': r['capital']['total_return_pct'],
                    'win_rate': r['trades']['win_rate_pct'],
                    'profit_factor': r['trades']['profit_factor'],
                    'max_dd': r['capital']['max_drawdown_pct'],
                    'total_trades': r['trades']['total'],
                }
                row.update(r['params'])
                data.append(row)
        
        df = pd.DataFrame(data)
        return df.sort_values('score', ascending=False)
