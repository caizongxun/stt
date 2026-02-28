"""
V1 Strategy - LightGBM Baseline
V1策略 - LightGBM基礎版本
"""
import streamlit as st
import joblib
from pathlib import Path
from core.gui_components import GUIComponents
from core.data_loader import DataLoader
from .trainer import Trainer
from .backtester import Backtester
from .config import V1Config
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

def render():
    st.header("🎯 V1 Strategy - LightGBM Baseline")
    st.info("""
    **特點:** 模型: LightGBM | 特點: 快速穩定 | 適用: 初學者 | 訓練時間: 2-5分鐘
    """)
    
    tab1, tab2, tab3 = st.tabs(["📊 訓練", "📈 回測", "📝 說明"])
    
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
        train_params = GUIComponents.render_training_params()
        st.markdown("**LightGBM參數**")
        num_leaves = st.slider("葉子節點數", 16, 128, 20, 1)
        max_depth = st.slider("最大深度", 3, 12, 4, 1)
        st.markdown("---")
        train_button = st.button("開始訓練", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 訓練過程")
        
        if train_button:
            with st.spinner("加載數據..."):
                loader = DataLoader()
                df = loader.load_klines(symbol, timeframe)
                st.success(f"[OK] 加載 {len(df)} 筆數據")
            
            config = V1Config(
                symbol=symbol,
                timeframe=timeframe,
                train_size=train_params['train_size'],
                num_leaves=num_leaves,
                max_depth=max_depth
            )
            
            trainer = Trainer(config)
            
            try:
                with st.spinner("訓練中..."):
                    results = trainer.train(df)
                
                st.success("✅ 訓練完成!")
                st.json(results)
                st.balloons()
            
            except Exception as e:
                st.error(f"訓練失敗: {e}")
                st.exception(e)
        else:
            st.info("""
            **使用步驟:**
            1. 左側選擇交易對和時間框架
            2. 調整訓練參數
            3. 點擊「開始訓練」
            4. 等待訓練完成
            5. 切換到「回測」頁面
            """)

def render_backtesting():
    st.subheader("策略回測")
    
    # 獲取所有模型
    models_dir = Path('models')
    if not models_dir.exists():
        st.warning("沒有找到模型，請先訓練模型")
        return
    
    model_paths = sorted([d for d in models_dir.iterdir() if d.is_dir() and 'v1' in d.name], reverse=True)
    
    if not model_paths:
        st.warning("沒有找到V1模型，請先訓練模型")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 回測參數")
        
        # 模型選擇
        model_names = [p.name for p in model_paths]
        selected_model = st.selectbox("選擇模型", model_names)
        
        st.markdown("---")
        
        # 回測參數
        backtest_params = GUIComponents.render_backtest_params()
        
        st.markdown("---")
        
        backtest_button = st.button("開始回測", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 回測結果")
        
        if backtest_button:
            model_path = models_dir / selected_model
            
            try:
                # 加載模型
                with st.spinner("加載模型..."):
                    model = joblib.load(model_path / 'model.pkl')
                    config_dict = joblib.load(model_path / 'config.pkl')
                    feature_names = joblib.load(model_path / 'features.pkl')
                    
                    # 更新回測參數
                    config_dict.update(backtest_params)
                    config = V1Config(**config_dict)
                    
                    st.success(f"✅ 模型加載完成: {selected_model}")
                
                # 加載數據
                with st.spinner("加載數據..."):
                    loader = DataLoader()
                    df = loader.load_klines(config.symbol, config.timeframe)
                    st.success(f"✅ 加載 {len(df)} 筆數據")
                
                # 執行回測
                with st.spinner("回測中..."):
                    backtester = Backtester(config)
                    results = backtester.run(model, df, feature_names)
                
                st.success("✅ 回測完成!")
                
                # 顯示結果
                display_backtest_results(results)
                
            except Exception as e:
                st.error(f"回測失敗: {e}")
                st.exception(e)
        else:
            st.info("""
            **使用步驟:**
            1. 選擇已訓練的模型
            2. 調整回測參數
            3. 點擊「開始回測」
            4. 查看回測結果
            """)

def display_backtest_results(results: dict):
    """顯示回測結果"""
    metrics = results['performance_metrics']
    
    # 關鍵指標
    st.markdown("### 📊 關鍵指標")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("總報酬", f"{metrics['total_return']:.2f} USDT")
        st.metric("報酬率", f"{metrics['total_return_pct']:.2f}%")
    
    with col2:
        st.metric("勝率", f"{metrics['win_rate']:.1f}%")
        st.metric("交易次數", f"{metrics['total_trades']}")
    
    with col3:
        st.metric("盈虧比", f"{metrics['profit_factor']:.2f}")
        st.metric("Sharpe比率", f"{metrics['sharpe_ratio']:.2f}")
    
    with col4:
        st.metric("最大回撤", f"{metrics['max_drawdown']:.2f}%")
        st.metric("最終資金", f"{metrics['final_capital']:.2f}")
    
    # 權益曲線
    st.markdown("### 💹 權益曲線")
    equity_data = results['equity_curve']
    if equity_data:
        df_equity = pd.DataFrame(equity_data)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_equity['timestamp'],
            y=df_equity['equity'],
            mode='lines',
            name='權益',
            line=dict(color='#00ff00', width=2)
        ))
        
        fig.update_layout(
            title='權益曲線',
            xaxis_title='時間',
            yaxis_title='權益 (USDT)',
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 詳細結果
    with st.expander("📊 查看詳細結果"):
        st.json(results)

def render_info():
    st.subheader("📝 V1策略說明")
    
    st.markdown("""
    ## 策略概述
    
    V1是基於 LightGBM 的基礎策略，適合初學者使用。
    
    ### 核心特點
    - **模型**: LightGBM
    - **特徵**: 25+技術指標
    - **標籤**: 三類別(做多/做空/不交易)
    - **訓練時間**: 2-5分鐘
    
    ### 優點
    - [+] 快速訓練
    - [+] 不需GPU
    - [+] 穩定可靠
    - [+] 易於調參
    
    ### 缺點
    - [-] 不具備時序學習
    - [-] 性能中等
    
    ### 預期效果
    - **月報酬**: 15-30%
    - **勝率**: 40-50%
    - **盈虧比**: 1.2-1.5
    - **最大回撤**: 15-25%
    
    ### 使用建議
    1. 先用BTCUSDT 15m訓練測試
    2. 確認回測效果良好再使用其他交易對
    3. 建議初始資金: 1000-10000 USDT
    4. 建議不使用槓桿或1x
    """)
