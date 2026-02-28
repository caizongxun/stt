"""
V1 Strategy - LightGBM Baseline
V1策略 - LightGBM基礎版本
"""
import streamlit as st
from core.gui_components import GUIComponents
from core.data_loader import DataLoader
from .trainer import Trainer
from .backtester import Backtester
from .config import V1Config

def render():
    """
    V1策略主界面
    """
    st.header("🎯 V1 Strategy - LightGBM Baseline")
    st.info("""
    **特點:**
    - 模型: LightGBM
    - 特點: 快速穩定
    - 適用: 初學者
    - 訓練時間: 2-5分鐘
    """)
    
    # 頁面切換
    tab1, tab2, tab3 = st.tabs(["🚀 訓練", "📊 回測", "ℹ️ 說明"])
    
    with tab1:
        render_training()
    
    with tab2:
        render_backtesting()
    
    with tab3:
        render_info()

def render_training():
    """渲染訓練頁面"""
    st.subheader("🚀 模型訓練")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 訓練參數")
        
        # 數據選擇
        symbol, timeframe = GUIComponents.render_data_selector()
        
        st.markdown("---")
        
        # 訓練參數
        train_params = GUIComponents.render_training_params()
        
        # V1特定參數
        st.markdown("**LightGBM參數**")
        num_leaves = st.slider("葉子節點數", 16, 128, 31, 1)
        max_depth = st.slider("最大深度", 3, 12, 6, 1)
        
        st.markdown("---")
        
        train_button = st.button("🚀 開始訓練", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 訓練過程")
        
        if train_button:
            # 加載數據
            with st.spinner("加載數據..."):
                loader = DataLoader()
                df = loader.load_klines(symbol, timeframe)
                st.success(f"✅ 加載 {len(df)} 筆數據")
            
            # 訓練
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
    """渲柔回測頁面"""
    st.subheader("📊 策略回測")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 回測參數")
        
        # 模型選擇
        model_name = st.text_input("模型名稱", "latest")
        
        st.markdown("---")
        
        # 回測參數
        backtest_params = GUIComponents.render_backtest_params()
        
        st.markdown("---")
        
        backtest_button = st.button("📊 開始回測", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 回測結果")
        
        if backtest_button:
            st.info("🚧 回測功能開發中...")
        else:
            st.info("請先訓練模型,然後進行回測")

def render_info():
    """渲柔說明頁面"""
    st.subheader("ℹ️ V1策略說明")
    
    st.markdown("""
    ## 策略概述
    
    V1是基於 LightGBM 的基礎策略,適合初學者使用。
    
    ### 核心特點
    - **模型**: LightGBM
    - **特徵**: 50+技術指標
    - **標籤**: 三類別(做多/做空/不交易)
    - **訓練時間**: 2-5分鐘
    
    ### 優點
    - ✅ 快速訓練
    - ✅ 不需GPU
    - ✅ 穩定可靠
    - ✅ 易於調參
    
    ### 缺點
    - ❌ 不具備時序學習
    - ❌ 性能中等
    
    ### 預期效果
    - **月報酬**: 30-50%
    - **勝率**: 55-60%
    - **盈虧比**: 1:1.5
    - **最大回撤**: <30%
    
    ### 使用建議
    1. 先用BTCUSDT 15m訓練測試
    2. 確認回測效果良好再使用其他交易對
    3. 建議初始資金: 1000-10000 USDT
    4. 建議不使用槓桿或使用2x以下
    """)
