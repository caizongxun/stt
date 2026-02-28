"""
V4 Strategy - Adaptive Dual Mode
V4策略 - 自適應雙模式
"""
import streamlit as st
import joblib
from pathlib import Path
from core.gui_components import GUIComponents
from core.data_loader import DataLoader
from .config import V4Config
from .trainer import V4Trainer
from .backtester import V4Backtester

def render():
    st.header("V4 Strategy - Adaptive Dual Mode")
    st.info("""
    **智能雙模式系統**
    
    - 盤整時: 區間反轉 (支撑買/壓力賣)
    - 趨勢時: 突破跟隨 (追隨動量)
    - 自動識別市場狀態並切換策略
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["訓練", "回測", "策略說明", "技術細節"])
    
    with tab1:
        render_training()
    
    with tab2:
        render_backtesting()
    
    with tab3:
        render_strategy_info()
    
    with tab4:
        render_technical_details()

def render_training():
    st.subheader("模型訓練")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 訓練參數")
        symbol, timeframe = GUIComponents.render_data_selector()
        
        st.markdown("---")
        st.markdown("**狀態識別**")
        adx_threshold = st.slider("ADX閉值", 15, 35, 25, 1)
        
        st.markdown("**區間參數**")
        range_window = st.slider("支撑/壓力窗口", 10, 30, 20, 2)
        range_target = st.slider("盤整目標", 0.3, 0.7, 0.5, 0.1)
        
        st.markdown("**突破參數**")
        volume_mult = st.slider("成交量倍數", 1.2, 2.5, 1.5, 0.1)
        
        st.markdown("---")
        train_button = st.button("開始訓練", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 訓練過程")
        
        if train_button:
            with st.spinner("加載數據..."):
                loader = DataLoader()
                df = loader.load_klines(symbol, timeframe)
                st.success(f"[OK] {len(df)} bars")
            
            config = V4Config(
                symbol=symbol,
                timeframe=timeframe,
                adx_ranging_threshold=adx_threshold,
                adx_trending_threshold=adx_threshold,
                support_resistance_window=range_window,
                range_target_pct=range_target,
                breakout_volume_multiplier=volume_mult
            )
            
            trainer = V4Trainer(config)
            
            try:
                with st.spinner("訓練中..."):
                    results = trainer.train(df)
                
                st.success("[OK] 訓練完成")
                
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Val AUC", f"{results['val_metrics']['auc']:.3f}")
                with col_b:
                    st.metric("Val精確", f"{results['val_metrics']['precision']*100:.1f}%")
                with col_c:
                    st.metric("OOS AUC", f"{results['oos_metrics']['auc']:.3f}")
                with col_d:
                    st.metric("OOS精確", f"{results['oos_metrics']['precision']*100:.1f}%")
                
                st.markdown("### 市場狀態分佈")
                col_e, col_f = st.columns(2)
                with col_e:
                    st.metric("盤整", f"{results['regime_statistics']['ranging_pct']:.1f}%")
                with col_f:
                    st.metric("趨勢", f"{results['regime_statistics']['trending_pct']:.1f}%")
                
                st.info(f"""
                **標籤統計:**
                - 正類率: {results['label_statistics']['positive_rate']:.1f}%
                - 盤整機會: {results['label_statistics']['ranging_positive']} ({results['label_statistics']['ranging_rate']:.1f}%)
                - 趨勢機會: {results['label_statistics']['trending_positive']} ({results['label_statistics']['trending_rate']:.1f}%)
                """)
                
                oos = results['oos_metrics']
                st.markdown("### OOS混淆矩陣")
                st.write(f"TN:{oos['tn']} FP:{oos['fp']} | FN:{oos['fn']} TP:{oos['tp']}")
                st.write(f"召回: {oos['recall']*100:.1f}%")
                
                with st.expander("Top特徵"):
                    for i, (name, imp) in enumerate(results['feature_importance'], 1):
                        st.write(f"{i}. {name}: {imp:.4f}")
                
                with st.expander("完整結果"):
                    st.json(results)
                
                if results['oos_metrics']['auc'] > 0.55:
                    st.success("OOS AUC > 0.55, 模型有預測能力")
            
            except Exception as e:
                st.error(f"訓練失敗: {e}")
                st.exception(e)
        else:
            st.info("""
            **V4訓練流程:**
            
            1. 識別市場狀態 (ADX+BB)
            2. 識別支撑/壓力
            3. 生成雙模式信號
            4. 生成自適應標籤
            5. 訓練集成模型
            
            **預期:**
            - 正類率: 25-35%
            - OOS AUC: 0.55-0.65
            - 盤整/趨勢各有優勢
            """)

def render_backtesting():
    st.subheader("策略回測")
    
    models_dir = Path('models')
    if not models_dir.exists():
        st.warning("沒有models資料夾,請先訓練模型")
        return
    
    v4_models = [d for d in models_dir.iterdir() if d.is_dir() and '_v4_' in d.name]
    
    if not v4_models:
        st.warning("沒有V4模型,請先訓練")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 回測參數")
        
        model_names = [m.name for m in sorted(v4_models, key=lambda x: x.name, reverse=True)]
        selected_model = st.selectbox("選擇模型", model_names)
        
        st.markdown("---")
        backtest_days = st.slider("回測天數", 7, 90, 30, 7)
        
        st.markdown("---")
        backtest_button = st.button("開始回測", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 回測結果")
        
        if backtest_button:
            model_path = models_dir / selected_model
            
            try:
                with st.spinner("加載模型..."):
                    models = []
                    for i in range(10):
                        model_file = model_path / f'model_{i}.pkl'
                        if model_file.exists():
                            models.append(joblib.load(model_file))
                    
                    config_dict = joblib.load(model_path / 'config.pkl')
                    feature_names = joblib.load(model_path / 'features.pkl')
                    st.success(f"[OK] {len(models)}個模型")
                
                with st.spinner("加載數據..."):
                    loader = DataLoader()
                    df = loader.load_klines(config_dict['symbol'], config_dict['timeframe'])
                    df = df.tail(backtest_days * 96)
                    st.success(f"[OK] {len(df)} bars")
                
                config = V4Config(**config_dict)
                backtester = V4Backtester(config)
                
                with st.spinner("執行回測..."):
                    results = backtester.run(models, df, feature_names)
                
                if results['status'] == 'no_trades':
                    st.warning(results['message'])
                else:
                    st.success("[OK] 回測完成")
                    
                    # 核心指標
                    st.markdown("### 核心結果")
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("總報酬", f"{results['capital']['total_return_pct']:.1f}%")
                    with col_b:
                        st.metric("勝率", f"{results['trades']['win_rate_pct']:.1f}%")
                    with col_c:
                        st.metric("利潤因子", f"{results['trades']['profit_factor']:.2f}")
                    with col_d:
                        st.metric("最大回撤", f"{results['capital']['max_drawdown_pct']:.1f}%")
                    
                    # 交易統計
                    st.markdown("### 交易統計")
                    col_e, col_f, col_g = st.columns(3)
                    with col_e:
                        st.metric("總交易", results['trades']['total'])
                    with col_f:
                        st.metric("獲利交易", results['trades']['winning'])
                    with col_g:
                        st.metric("虧損交易", results['trades']['losing'])
                    
                    # 分狀態表現
                    st.markdown("### 分狀態表現")
                    regime = results['regime_performance']
                    col_h, col_i = st.columns(2)
                    with col_h:
                        st.info(f"""
                        **盤整模式:**
                        - 交易: {regime['ranging_trades']}
                        - 勝率: {regime['ranging_win_rate']:.1f}%
                        - 盈虧: {regime['ranging_pnl']:.2f}
                        """)
                    with col_i:
                        st.info(f"""
                        **趨勢模式:**
                        - 交易: {regime['trending_trades']}
                        - 勝率: {regime['trending_win_rate']:.1f}%
                        - 盈虧: {regime['trending_pnl']:.2f}
                        """)
                    
                    # 出場原因
                    with st.expander("出場原因分佈"):
                        st.json(results['exit_reasons'])
                    
                    # 近期交易
                    with st.expander("近期交易示例"):
                        for trade in results['trades_sample']:
                            st.write(trade)
                    
                    # 完整結果
                    with st.expander("完整JSON"):
                        st.json(results)
            
            except Exception as e:
                st.error(f"回測失敗: {e}")
                st.exception(e)
        else:
            st.info("選擇模型並點擊回測")

def render_strategy_info():
    st.subheader("策略說明")
    
    st.markdown("""
    ## V4自適應雙模式
    
    ### 核心概念
    
    市場只有兩種狀態:
    1. **盤整** (70%時間) - 區間反轉
    2. **趨勢** (30%時間) - 突破跟隨
    
    用ML識別當前狀態,自動切換策略。
    
    ### 盤整模式
    
    **識別條件:**
    - ADX < 25 (趨勢弱)
    - BB寬度 < 25%分位 (低波動)
    
    **交易策略:**
    - 支撑附近 + RSI<30 -> 做多
    - 壓力附近 + RSI>70 -> 做空
    - 目標: 區間的50%
    
    ### 突破模式
    
    **識別條件:**
    - ADX > 25 (趨勢強)
    
    **交易策略:**
    - 突破壓力 + 成交量放大 -> 做多
    - 跌破支撑 + 成交量放大 -> 做空
    - 目標: 3 ATR
    
    ### 優勢
    
    1. **自適應** - 不同市場用不同策略
    2. **互補** - 盤整高勝率,突破高盈虧比
    3. **可解釋** - 每個信號有明確邏輯
    4. **符合人性** - 基於真實市場行為
    """)

def render_technical_details():
    st.subheader("技術細節")
    
    st.markdown("""
    ### 市場狀態識別
    
    ```python
    # 盤整檢測
    if ADX < 25 and BB寬度 < 25%分位:
        regime = RANGING
    
    # 趨勢檢測
    if ADX >= 25:
        regime = TRENDING
    ```
    
    ### 支撑/壓力識別
    
    ```python
    support = low.rolling(20).min()
    resistance = high.rolling(20).max()
    
    near_support = abs(close - support) / close < 2%
    near_resistance = abs(close - resistance) / close < 2%
    ```
    
    ### 標籤生成
    
    **盤整模式:**
    ```python
    target = support + range_width * 0.5
    label = 1 if future_high >= target else 0
    ```
    
    **突破模式:**
    ```python
    target = close + 3 * ATR
    label = 1 if future_high >= target else 0
    ```
    
    ### 特徵集
    
    - 狀態特徵: ADX, BB寬度, 狀態編碼
    - 結構特徵: 區間位置, 距離支撑/壓力
    - 技術指標: RSI, MACD, ATR
    - 信號特徵: 做多/做空信號
    """)
