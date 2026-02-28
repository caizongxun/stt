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
    st.header("V2.2 Strategy - BB Reversal (Balanced)")
    st.info("""
    **V2.2:** 無歷史特徵 | 嚴格標籤(2.5 ATR) | 平衡閉值(0.55)
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
        
        st.markdown("**標籤參數**")
        min_reversal_atr = st.slider("最小反轉ATR", 1.5, 3.5, 2.5, 0.1)
        reversal_lookforward = st.slider("反轉觀察周期", 10, 30, 15, 1)
        
        st.markdown("**預測閉值**")
        predict_threshold = st.slider("閉值", 0.50, 0.80, 0.55, 0.01)
        st.caption("推薦: 0.55")
        
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
                min_reversal_atr=min_reversal_atr,
                reversal_lookforward=reversal_lookforward,
                predict_threshold=predict_threshold
            )
            
            trainer = Trainer(config)
            
            try:
                with st.spinner("生成標籤..."):
                    st.info("步驟1: 檢測BB觸碰...")
                    st.info("步驟2: 識別有效反轉 (2.5 ATR)...")
                    st.info("步驟3: 生成特徵 (無歷史特徵)...")
                
                with st.spinner("訓練模型..."):
                    results = trainer.train(df)
                
                st.success("[OK] 訓練完成!")
                
                # 顯示結果
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("驗證準確率", f"{results['val_metrics']['accuracy']*100:.1f}%")
                with col_b:
                    st.metric("驗證精確率", f"{results['val_metrics']['precision']*100:.1f}%")
                with col_c:
                    st.metric("驗證召回率", f"{results['val_metrics']['recall']*100:.1f}%")
                with col_d:
                    st.metric("驗證AUC", f"{results['val_metrics']['auc']:.3f}")
                
                st.info(f"""
                **標籤統計:**
                - BB觸碰: {results['label_statistics']['total_touches']}
                - 有效反轉: {results['label_statistics']['valid_reversals']}
                - 反轉率: {results['label_statistics']['reversal_rate']:.1f}%
                
                **混淆矩陣:**
                - TN: {results['val_metrics']['true_negative']} | FP: {results['val_metrics']['false_positive']}
                - FN: {results['val_metrics']['false_negative']} | TP: {results['val_metrics']['true_positive']}
                """)
                
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
            **V2.2 參數:**
            - 最小反轉ATR: 2.5
            - 觀察周期: 15根K棒
            - 預測閉值: 0.55
            
            **預期:**
            - AUC: 0.60-0.65
            - 精確率50-60%
            - 召回率30-40%
            """)

def render_backtesting():
    st.subheader("策略回測")
    
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
                    st.success(f"[OK] 模型加載 (閉值: {config_dict.get('predict_threshold', 0.5)})")
                
                with st.spinner("加載數據..."):
                    loader = DataLoader()
                    df = loader.load_klines(config_dict['symbol'], config_dict['timeframe'])
                    df = df.tail(backtest_days * 96)
                    st.success(f"[OK] 加載 {len(df)} 筆數據")
                
                config = V2Config(**config_dict)
                backtester = Backtester(config)
                
                with st.spinner("執行回測..."):
                    results = backtester.run(model, df, feature_names)
                
                if results['status'] == 'success':
                    st.success("[OK] 回測完成!")
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("總報酬", f"{results['capital']['total_return']:.2f}%")
                    with col_b:
                        st.metric("勝率", f"{results['trades']['win_rate']:.1f}%")
                    with col_c:
                        st.metric("盈虧因子", f"{results['profit']['profit_factor']:.2f}")
                    with col_d:
                        st.metric("最大回撤", f"{results['capital']['max_drawdown']:.2f}%")
                    
                    st.markdown("### 交易統計")
                    col_e, col_f = st.columns(2)
                    with col_e:
                        st.write(f"總交易: {results['trades']['total']}筆")
                        st.write(f"獲利: {results['trades']['winning']}筆")
                        st.write(f"虧損: {results['trades']['losing']}筆")
                    with col_f:
                        st.write(f"平均獲利: ${results['profit']['avg_win']:.2f}")
                        st.write(f"平均虧損: ${results['profit']['avg_loss']:.2f}")
                        st.write(f"持倉: {results['trades']['avg_bars_held']:.1f}根K棒")
                    
                    with st.expander("詳細結果"):
                        st.json(results)
                    
                    if results['capital']['total_return'] > 2:
                        st.balloons()
                else:
                    st.warning(results.get('message', '回測失敗'))
            
            except Exception as e:
                st.error(f"回測失敗: {e}")
                st.exception(e)
        else:
            st.info("**預期:** 50-100筆/90天 | 勝率45%+ | 報酬3-8%")

def render_info():
    st.subheader("V2.2 說明")
    st.markdown("""
    **V2.2優化:**
    - 最小反轉: 2.5 ATR
    - 觀察周期: 15根K棒
    - 閉值: 0.55 (平衡點)
    - 正類權重: 3.0
    """)
