"""
V2 Strategy - BB Reversal System
V2策略 - BB反轉系統
"""
import streamlit as st
import joblib
from pathlib import Path
from core.gui_components import GUIComponents
from core.data_loader import DataLoader
from .trainer import Trainer
from .backtester import Backtester
from .config import V2Config
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

def render():
    st.header("V2 Strategy - BB Reversal System")
    st.info("""
    **特點:** BB反轉預測 | ATR動態風控 | 分批進場 | 適用: 進階使用者
    """)
    
    tab1, tab2, tab3 = st.tabs(["訓練", "回測", "說明"])
    
    with tab1:
        render_training()
    
    with tab2:
        render_backtesting()
    
    with tab3:
        render_info()

def render_training():
    st.subheader("模型訓練")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 訓練參數")
        symbol, timeframe = GUIComponents.render_data_selector()
        st.markdown("---")
        
        st.markdown("**BB參數**")
        bb_window = st.slider("BB周期", 10, 30, 20, 1)
        bb_std = st.slider("BB標準差", 1.5, 3.0, 2.0, 0.1)
        
        st.markdown("**ATR參數**")
        atr_window = st.slider("ATR周期", 10, 20, 14, 1)
        atr_sl_multiplier = st.slider("止損倍數", 1.5, 3.0, 2.0, 0.1)
        atr_tp_multiplier = st.slider("止盈倍數", 2.0, 5.0, 3.0, 0.1)
        
        st.markdown("---")
        train_params = GUIComponents.render_training_params()
        
        st.markdown("---")
        train_button = st.button("開始訓練", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 訓練過程")
        
        if train_button:
            with st.spinner("加載數據..."):
                loader = DataLoader()
                df = loader.load_klines(symbol, timeframe)
                st.success(f"[OK] 加載 {len(df)} 筆數據")
            
            config = V2Config(
                symbol=symbol,
                timeframe=timeframe,
                train_size=train_params['train_size'],
                bb_window=bb_window,
                bb_std=bb_std,
                atr_window=atr_window,
                atr_sl_multiplier=atr_sl_multiplier,
                atr_tp_multiplier=atr_tp_multiplier
            )
            
            trainer = Trainer(config)
            
            try:
                with st.spinner("生成標籤..."):
                    st.info("步驟1: 檢測BB觸碰...")
                    st.info("步驟2: 識別有效反轉...")
                    st.info("步驟3: 生成特徵...")
                
                with st.spinner("訓練模型..."):
                    results = trainer.train(df)
                
                st.success("[OK] 訓練完成!")
                st.json(results)
                st.balloons()
            
            except Exception as e:
                st.error(f"訓練失敗: {e}")
                st.exception(e)
        else:
            st.info("""
            **V2策略特點:**
            
            1. 只在BB觸碰時預測反轉
            2. ATR動態止盈止損
            3. 智能分批進場
            4. 高勝率低頻交易
            
            **使用步驟:**
            1. 調整BB和ATR參數
            2. 點擊「開始訓練」
            3. 等待訓練完成
            4. 切換到「回測」
            """)

def render_backtesting():
    st.subheader("策略回測")
    st.warning("回測功能開發中...")

def render_info():
    st.subheader("V2策略說明")
    
    st.markdown("""
    ## 策略概述
    
    V2是基於Bollinger Bands反轉預測的進階策略。
    
    ### 核心理念
    
    不預測市場方向,而是預測**BB觸碰後是否會有效反轉**。
    
    ### 特點
    
    - **精準進場**: 只在BB觸碰時交易
    - **ATR風控**: 止盈止損隨市場波動調整
    - **分批進場**: 降低單次進場風險
    - **高勝率**: 目標勝率60-70%
    
    ### 預期效果
    
    - **月報酬**: 10-20%
    - **勝率60-70%
    - **盈虧比1.5-2.0
    - **最大回撤10-15%
    - **月交易**: 10-30筆
    
    ### 適用場景
    
    - 震盪市場
    - 波動率中等市場
    - 有明顯支撐/壓力位
    
    ### 不適用場景
    
    - 強勢單邊走勢
    - 極端低波動率
    - 重大消息面前後
    """)
