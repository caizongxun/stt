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
        
        st.markdown("**預測閉值**")
        predict_threshold = st.slider("閉值", 0.5, 0.9, 0.6, 0.05)
        
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
                atr_tp_multiplier=atr_tp_multiplier,
                predict_threshold=predict_threshold
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
                
                # 顯示結果
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("驗證準確率", f"{results['val_metrics']['accuracy']*100:.1f}%")
                with col_b:
                    st.metric("驗證精確率", f"{results['val_metrics']['precision']*100:.1f}%")
                with col_c:
                    st.metric("驗證AUC", f"{results['val_metrics']['auc']:.3f}")
                
                with st.expander("詳細結果"):
                    st.json(results)
                
                with st.expander("Top 10 特徵"):
                    for i, (name, imp) in enumerate(results['top_10_features'], 1):
                        st.write(f"{i}. **{name}**: {imp}")
                
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
    
    # 選擇模型
    models_dir = Path('models')
    if not models_dir.exists():
        st.warning("沒有找到models資料夾,請先訓練模型")
        return
    
    v2_models = [d for d in models_dir.iterdir() if d.is_dir() and '_v2_' in d.name]
    
    if not v2_models:
        st.warning("沒有V2模型,請先訓練")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 回測參數")
        
        model_names = [m.name for m in sorted(v2_models, key=lambda x: x.name, reverse=True)]
        selected_model = st.selectbox("選擇模型", model_names)
        
        st.markdown("---")
        backtest_days = st.slider("回測天數", 7, 180, 90, 7)
        
        st.markdown("---")
        backtest_button = st.button("開始回測", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 回測結果")
        
        if backtest_button:
            model_path = models_dir / selected_model
            
            try:
                with st.spinner("加載模型..."):
                    model = joblib.load(model_path / 'model.pkl')
                    config_dict = joblib.load(model_path / 'config.pkl')
                    feature_names = joblib.load(model_path / 'features.pkl')
                    st.success("[OK] 模型加載完成")
                
                with st.spinner("加載數據..."):
                    loader = DataLoader()
                    df = loader.load_klines(config_dict['symbol'], config_dict['timeframe'])
                    df = df.tail(backtest_days * 96)  # 15m: 96根/天
                    st.success(f"[OK] 加載 {len(df)} 筆數據")
                
                # 重建配置
                config = V2Config(**config_dict)
                backtester = Backtester(config)
                
                with st.spinner("執行回測..."):
                    results = backtester.run(model, df, feature_names)
                
                if results['status'] == 'success':
                    st.success("[OK] 回測完成!")
                    
                    # 核心指標
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("總報酬", f"{results['capital']['total_return']:.2f}%")
                    with col_b:
                        st.metric("勝率", f"{results['trades']['win_rate']:.1f}%")
                    with col_c:
                        st.metric("盈虧因子", f"{results['profit']['profit_factor']:.2f}")
                    with col_d:
                        st.metric("最大回撤", f"{results['capital']['max_drawdown']:.2f}%")
                    
                    # 交易統計
                    st.markdown("### 交易統計")
                    col_e, col_f = st.columns(2)
                    with col_e:
                        st.write(f"總交易: {results['trades']['total']}筆")
                        st.write(f"獲利交易: {results['trades']['winning']}筆")
                        st.write(f"虧損交易: {results['trades']['losing']}筆")
                    with col_f:
                        st.write(f"平均獲利: ${results['profit']['avg_win']:.2f}")
                        st.write(f"平均虧損: ${results['profit']['avg_loss']:.2f}")
                        st.write(f"平均持倉: {results['trades']['avg_bars_held']:.1f}根K棒")
                    
                    # 方向統計
                    st.markdown("### 方向統計")
                    col_g, col_h = st.columns(2)
                    with col_g:
                        st.write(f"LONG交易: {results['direction']['long_trades']}筆")
                        st.write(f"LONG勝率: {results['direction']['long_win_rate']:.1f}%")
                    with col_h:
                        st.write(f"SHORT交易: {results['direction']['short_trades']}筆")
                        st.write(f"SHORT勝率: {results['direction']['short_win_rate']:.1f}%")
                    
                    # 平倉原因
                    st.markdown("### 平倉原因")
                    st.write(results['exit_reasons'])
                    
                    # 詳細結果
                    with st.expander("詳細結果"):
                        st.json(results)
                    
                    with st.expander("最近10筆交易"):
                        st.dataframe(pd.DataFrame(results['trades_detail']))
                    
                    st.balloons()
                else:
                    st.warning(results.get('message', '回測失敗'))
            
            except Exception as e:
                st.error(f"回測失敗: {e}")
                st.exception(e)
        else:
            st.info("""
            **回測說明:**
            
            1. 選擇已訓練的模型
            2. 設定回測天數
            3. 點擊「開始回測」
            
            **回測特點:**
            - ATR動態止盈止損
            - 每日最套5筆交易
            - 2%單筆風險控制
            - 模擬真實手續費
            """)

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
    - **高勝率**: 目標勝率56%+
    
    ### 預期效果
    
    - **月報酬**: 10-20%
    - **勝率**: 56%+
    - **盈虧比**: 1.5-2.0
    - **最大回撤**: 10-15%
    - **月交易**: 30-50筆
    
    ### 適用場景
    
    - 震盪市場
    - 波動率中等市場
    - 有明顯支撐/壓力位
    
    ### 不適用場景
    
    - 強勢單邊走勢
    - 極端低波動率
    - 重大消息面前後
    """)
