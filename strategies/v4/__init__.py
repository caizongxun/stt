"""
V4 Strategy - Adaptive Dual Mode
V4策略 - 自適應雙模式
"""
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
from core.gui_components import GUIComponents
from core.data_loader import DataLoader
from .config import V4Config
from .trainer import V4Trainer
from .backtester import V4Backtester
from .optimizer import ParameterOptimizer

def render():
    st.header("V4 Strategy - Adaptive Dual Mode")
    st.info("""
    **智能雙模式系統**
    
    - 盤整時: 區間反轉 (支撑買/壓力賣)
    - 趨勢時: 突破跟隨 (追隨動量)
    - 自動識別市場狀態並切換策略
    """)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["訓練", "回測", "參數優化", "策略說明", "技術細節"])
    
    with tab1:
        render_training()
    
    with tab2:
        render_backtesting()
    
    with tab3:
        render_optimization()
    
    with tab4:
        render_strategy_info()
    
    with tab5:
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
                st.write(f"TN:{oos['tn']} FP:{oos['fp']} | FN:{oss['fn']} TP:{oos['tp']}")
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
    """(略,保持原樣)"""
    pass

def render_optimization():
    st.subheader("參數優化")
    st.warning("優化過程需褁1-3分鐘,請耐心等待")
    
    models_dir = Path('models')
    if not models_dir.exists():
        st.warning("沒有models資料夾")
        return
    
    v4_models = [d for d in models_dir.iterdir() if d.is_dir() and '_v4_' in d.name]
    if not v4_models:
        st.warning("沒有V4模型")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 優化設置")
        
        model_names = [m.name for m in sorted(v4_models, key=lambda x: x.name, reverse=True)]
        selected_model = st.selectbox("選擇模型", model_names)
        
        opt_days = st.slider("優化天數", 14, 60, 30, 7)
        
        st.markdown("**搜索範圍**")
        use_default = st.checkbox("使用預設範圍", value=True)
        
        if not use_default:
            st.info("自定義範圍功能開發中")
        
        optimize_button = st.button("開始優化", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 優化結果")
        
        if optimize_button:
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
                    df = df.tail(opt_days * 96)
                    st.success(f"[OK] {len(df)} bars")
                
                config = V4Config(**config_dict)
                optimizer = ParameterOptimizer(config)
                
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                with st.spinner("執行參數優化..."):
                    opt_results = optimizer.optimize(models, df, feature_names)
                
                st.success("[OK] 優化完成!")
                
                best = opt_results['best_result']
                
                st.markdown("### 最佳參數")
                st.json(opt_results['best_params'])
                
                st.markdown("### 最佳表現")
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("總報酬", f"{best['capital']['total_return_pct']:.1f}%")
                with col_b:
                    st.metric("勝率", f"{best['trades']['win_rate_pct']:.1f}%")
                with col_c:
                    st.metric("利潤因子", f"{best['trades']['profit_factor']:.2f}")
                with col_d:
                    st.metric("最大回撤", f"{best['capital']['max_drawdown_pct']:.1f}%")
                
                st.markdown("### Top 10參數組合")
                report_df = optimizer.get_optimization_report()
                st.dataframe(report_df.head(10))
                
                with st.expander("完整優化報告"):
                    st.dataframe(report_df)
                
                with st.expander("最佳結果詳情"):
                    st.json(best)
            
            except Exception as e:
                st.error(f"優化失敗: {e}")
                st.exception(e)
        else:
            st.info("""
            **優化說明:**
            
            系統將測試多個參數組合:
            - 預測閉值: 0.45, 0.50, 0.55
            - 止損: 1.0, 1.2, 1.5 ATR
            - 盤整止盈: 2.0, 2.5, 3.0 ATR
            - 趨勢止盈: 3.0, 4.0, 5.0 ATR
            - 槓桶: 2x, 3x, 4x
            - 仓位: 25%, 30%, 35%
            
            **總組合: 3^6 = 729種**
            
            評分標準:
            - 40% 報酬率
            - 20% 勝率
            - 20% 利潤因子
            - 20% 回撤控制
            """)

def render_strategy_info():
    """(略,保持原樣)"""
    pass

def render_technical_details():
    """(略,保持原樣)"""
    pass
