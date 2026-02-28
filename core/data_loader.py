"""
Unified Data Loader for HuggingFace Dataset
統一數據加載器
"""
from huggingface_hub import hf_hub_download
import pandas as pd
from pathlib import Path
import streamlit as st

class DataLoader:
    """
    統一數據加載器
    支持從HuggingFace加載K線數據
    """
    
    REPO_ID = "zongowo111/v2-crypto-ohlcv-data"
    
    SYMBOLS = [
        'AAVEUSDT', 'ADAUSDT', 'ALGOUSDT', 'ARBUSDT', 'ATOMUSDT',
        'AVAXUSDT', 'BALUSDT', 'BATUSDT', 'BCHUSDT', 'BNBUSDT',
        'BTCUSDT', 'COMPUSDT', 'CRVUSDT', 'DOGEUSDT', 'DOTUSDT',
        'ENJUSDT', 'ENSUSDT', 'ETCUSDT', 'ETHUSDT', 'FILUSDT',
        'GALAUSDT', 'GRTUSDT', 'IMXUSDT', 'KAVAUSDT', 'LINKUSDT',
        'LTCUSDT', 'MANAUSDT', 'MATICUSDT', 'MKRUSDT', 'NEARUSDT',
        'OPUSDT', 'SANDUSDT', 'SNXUSDT', 'SOLUSDT', 'SPELLUSDT',
        'UNIUSDT', 'XRPUSDT', 'ZRXUSDT'
    ]
    
    TIMEFRAMES = ['1m', '15m', '1h', '1d']
    
    def __init__(self):
        self.cache_dir = Path('data/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @st.cache_data(ttl=3600)
    def load_klines(_self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        加載K線數據
        
        Args:
            symbol: 交易對,例如 'BTCUSDT'
            timeframe: '1m', '15m', '1h', '1d'
        
        Returns:
            pd.DataFrame: K線數據
        """
        base = symbol.replace("USDT", "")
        filename = f"{base}_{timeframe}.parquet"
        path_in_repo = f"klines/{symbol}/{filename}"
        
        try:
            local_path = hf_hub_download(
                repo_id=_self.REPO_ID,
                filename=path_in_repo,
                repo_type="dataset",
                cache_dir=str(_self.cache_dir)
            )
            df = pd.read_parquet(local_path)
            
            # FIX: 轉換所有數值欄位為數值型別
            numeric_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 處理ignore欄位
            if 'ignore' in df.columns:
                df['ignore'] = pd.to_numeric(df['ignore'], errors='coerce')
            
            # 確保open_time和close_time是時間型別
            if 'open_time' in df.columns:
                df['open_time'] = pd.to_datetime(df['open_time'])
            if 'close_time' in df.columns:
                df['close_time'] = pd.to_datetime(df['close_time'])
            
            return df
            
        except Exception as e:
            st.error(f"數據加載失敗: {e}")
            return pd.DataFrame()
    
    def get_data_info(self, symbol: str, timeframe: str) -> dict:
        """
        獲取數據信息
        """
        df = self.load_klines(symbol, timeframe)
        if df.empty:
            return {}
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'rows': len(df),
            'start_date': df['open_time'].min(),
            'end_date': df['open_time'].max(),
            'columns': list(df.columns)
        }
