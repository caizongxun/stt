"""
V3 Strategy - Aggressive High Performance
V3策略 - 激進高性能系統
"""
import streamlit as st
import joblib
from pathlib import Path
from core.gui_components import GUIComponents
from core.data_loader import DataLoader
from .config import V3Config
from .trainer import EnsembleTrainer
from .backtester import AggressiveBacktester

def render():
    st.header("V3 Strategy - High Performance (50% in 30 days)")
    st.warning("""
    **激進策略警告**
    
    - 5倍槓桶 | 30%仓位 | 高頻交易
    - 目標30天報酬 50%
    - 最大回撤 20%
    - 僅適用於風險承受能力高的交易者
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["訓練", "回測", "策略說明", "風險揭露"])
    
    with tab1:
        render_training()
    
    with tab2:
        render_backtesting()
    
    with tab3:
        render_strategy_info()
    
    with tab4:
        render_risk_disclosure()

def render_training():
    st.subheader("模型訓練")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 訓練參數")
        symbol, timeframe = GUIComponents.render_data_selector()
        
        st.markdown("---")
        st.markdown("**策略組合**")
        use_bb = st.checkbox("BB反轉", value=True)
        use_momentum = st.checkbox("動量突破", value=True)
        use_trend = st.checkbox("趨勢跟隨", value=True)
        
        st.markdown("**資金管理**")
        leverage = st.slider("槓桶倍數", 1, 10, 5, 1)
        position_pct = st.slider("仓位比例", 0.1, 0.5, 0.3, 0.05)
        use_compound = st.checkbox("複利模式", value=True)
        
        st.markdown("**集成學習**")
        use_ensemble = st.checkbox("啟用集成", value=True)
        n_models = st.slider("模型數量", 1, 10, 5, 1) if use_ensemble else 1
        
        st.markdown("---")
        train_button = st.button("開始訓練", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 訓練過程")
        
        if train_button:
            with st.spinner("加載數據..."):
                loader = DataLoader()
                df = loader.load_klines(symbol, timeframe)
                st.success(f"[OK] 加載 {len(df)} 筆數據")
            
            config = V3Config(
                symbol=symbol,
                timeframe=timeframe,
                use_bb_reversal=use_bb,
                use_momentum_breakout=use_momentum,
                use_trend_following=use_trend,
                leverage=leverage,
                position_pct=position_pct,
                use_compound=use_compound,
                use_ensemble=use_ensemble,
                ensemble_models=n_models
            )
            
            trainer = EnsembleTrainer(config)
            
            try:
                with st.spinner("訓練中... (預計1-2分鐘)"):
                    results = trainer.train(df)
                
                st.success("[OK] 訓練完成!")
                
                # 顯示結果
                st.markdown("### 模型指標")
                
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Val AUC", f"{results['val_metrics']['auc']:.3f}")
                with col_b:
                    st.metric("Val精確率", f"{results['val_metrics']['precision']*100:.1f}%")
                with col_c:
                    st.metric("OOS AUC", f"{results['oos_metrics']['auc']:.3f}")
                with col_d:
                    st.metric("OOS精確率", f"{results['oos_metrics']['precision']*100:.1f}%")
                
                st.info(f"""
                **標籤統計:**
                - 正類率: {results['label_statistics']['positive_rate']:.1f}%
                - LONG標籤: {results['label_statistics']['long_labels']}
                - SHORT標籤: {results['label_statistics']['short_labels']}
                
                **OOS驗證:**
                - Train: {results['split_info']['train_bars']} ({results['split_info']['train_pct']:.1f}%)
                - Val: {results['split_info']['val_bars']} ({results['split_info']['val_pct']:.1f}%)
                - OOS: {results['split_info']['oos_bars']} ({results['split_info']['oos_pct']:.1f}%)
                - 無洩漏: {results['validation_check']['is_valid']}
                """)
                
                # OOS混淆矩陣
                oos_m = results['oos_metrics']
                st.markdown("### OOS混淆矩陣 (關鍵)")
                st.write(f"TN: {oos_m['tn']} | FP: {oos_m['fp']}")
                st.write(f"FN: {oos_m['fn']} | TP: {oos_m['tp']}")
                st.write(f"召回率: {oos_m['recall']*100:.1f}%")
                
                with st.expander("Top 10特徵"):
                    for i, (name, imp) in enumerate(results['feature_importance'][:10], 1):
                        st.write(f"{i}. {name}: {imp:.4f}")
                
                with st.expander("完整結果"):
                    st.json(results)
                
                if results['oos_metrics']['auc'] > 0.6:
                    st.balloons()
                    st.success("OOS AUC > 0.6, 模型泛化能力良好!")
            
            except Exception as e:
                st.error(f"訓練失敗: {e}")
                st.exception(e)
        else:
            st.info("""
            **V3訓練流程:**
            
            1. 生成多策略信號
            2. 生成智能標籤 (嚴格)
            3. OOS分割 (75:12.5:12.5)
            4. 特徵工程 (50+)
            5. 集成訓練 (XGBoost)
            6. OOS驗證
            
            **預期指標:**
            - 正類率: 20-30%
            - OOS AUC: 0.60+
            - OOS精確率: 50%+
            - OOS召回率: 40%+
            """)

def render_backtesting():
    st.subheader("策略回測")
    
    models_dir = Path('models')
    if not models_dir.exists():
        st.warning("沒有找到models資料夾,請先訓練模型")
        return
    
    v3_models = [d for d in models_dir.iterdir() if d.is_dir() and '_v3_' in d.name]
    
    if not v3_models:
        st.warning("沒有V3模型,請先訓練")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 回測參數")
        
        model_names = [m.name for m in sorted(v3_models, key=lambda x: x.name, reverse=True)]
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
                    st.success(f"[OK] 加載 {len(models)} 個模型")
                
                with st.spinner("加載數據..."):
                    loader = DataLoader()
                    df = loader.load_klines(config_dict['symbol'], config_dict['timeframe'])
                    df = df.tail(backtest_days * 96)
                    st.success(f"[OK] 加載 {len(df)} 筆數據")
                
                config = V3Config(**config_dict)
                backtester = AggressiveBacktester(config)
                
                with st.spinner("執行回測..."):
                    results = backtester.run(models, df, feature_names)
                
                st.success("[OK] 回測完成!")
                st.json(results)
            
            except Exception as e:
                st.error(f"回測失敗: {e}")
                st.exception(e)
        else:
            st.info("選擇模型並點擊回測")

def render_strategy_info():
    st.subheader("策略說明")
    
    st.markdown("""
    ## V3高性能系統
    
    ### 目標
    30天報酬率 50%
    
    ### 核心特色
    
    1. 多策略融合
       - BB反轉
       - 動量突破
       - 趨勢跟隨
    
    2. 嚴格OOS驗證
       - Train 75%
       - Val 12.5%
       - OOS 12.5%
    
    3. 集成學習
       - 5個XGBoost
       - 投票機制
    
    4. 激進資金
       - 5倍槓桶
       - 30%仓位
       - 複利滾動
    """)

def render_risk_disclosure():
    st.subheader("風險揭露")
    
    st.error("""
    ### 重要風險提示
    
    1. 高槓桶風險
       - 5倍槓桶意味盈虧和虧損都放大5個
    
    2. 高頻交易風險
       - 手續費累積
       - 滑點影響
    
    3. 回測與實盤差異
       - 理想執行假設
    
    4. 使用建議
       - 只用可損失資金
       - 先小資金測試
    """)
