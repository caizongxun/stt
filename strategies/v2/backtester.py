"""
V2 Backtester - Placeholder
V2回測模組 - 占位符
"""
import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, config):
        self.config = config
    
    def run(self, model, df: pd.DataFrame, feature_names: list) -> dict:
        """
        回測 - 待實現
        """
        return {
            'status': 'not_implemented',
            'message': '回測功能開發中'
        }
