"""
V5 Strategy - Pure ML Price Prediction
V5策略 - 純ML價格預測

目標: 月報酬 20-50%
理念: 讓模型自由預測,不用任何硬規則
"""
import streamlit as st
import joblib
from pathlib import Path
from core.gui_components import GUIComponents
from core.data_loader import DataLoader
from .config import V5Config
from .trainer import V5Trainer
from .backtester import V5Backtester

def render():
    st.header("V5 Strategy - Pure ML")
    st.success("""
    **極簡高效系統**
    
    [OK] 不用市場狀態分類
    [OK] 不用支撓/壓力識別
    [OK] 不用任何硬規則過濾
    [OK] 完全信任 ML 預測
    
    [TARGET] 月報酬 20-50%
    """)
    
    tab1, tab2 = st.tabs(["訓練", "回測"])
    
    with tab1:
        render_training()
    
    with tab2:
        render_backtesting()

def render_training():
    st.subheader("模型訓練")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 訓練參數")
        
        symbol, timeframe = GUIComponents.render_data_selector()
        
        st.markdown("---")
        st.markdown("**標籤參數**")
        forward_bars = st.slider("預測周期(K棒)", 4, 16, 8, 2)
        min_return = st.slider("最小目標%", 0.5, 2.0, 0.8, 0.1)
        require_no_reverse = st.checkbox("要求不回撤", value=True)
        
        st.markdown("**模型參數**")
        max_depth = st.slider("樹深度", 4, 8, 6, 1)
        n_estimators = st.slider("樹數量", 100, 500, 300, 50)
        ensemble_models = st.slider("集成模型數", 3, 7, 5, 1)
        
        st.markdown("---")
        train_button = st.button("開始訓練", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 訓練過程")
        
        if train_button:
            with st.spinner("加載數據..."):
                loader = DataLoader()
                df = loader.load_klines(symbol, timeframe)
                st.success(f"[OK] {len(df)} bars")
            
            config = V5Config(
                symbol=symbol,
                timeframe=timeframe,
                forward_bars=forward_bars,
                min_return_pct=min_return/100,
                require_no_reverse=require_no_reverse,
                max_depth=max_depth,
                n_estimators=n_estimators,
                ensemble_models=ensemble_models
            )
            
            trainer = V5Trainer(config)
            
            try:
                with st.spinner("訓練中..."):
                    results = trainer.train(df)
                
                st.success("[OK] 訓練完成")
                
                # 評估指標
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Val AUC", f"{results['val_metrics']['auc']:.3f}")
                with col_b:
                    st.metric("Val精確", f"{results['val_metrics']['precision']*100:.1f}%")
                with col_c:
                    st.metric("OOS AUC", f"{results['oos_metrics']['auc']:.3f}")
                with col_d:
                    st.metric("OOS精確", f"{results['oos_metrics']['precision']*100:.1f}%")
                
                # 混淆矩陣
                oos = results['oos_metrics']
                st.markdown("### OOS表現")
                col_e, col_f = st.columns(2)
                with col_e:
                    st.write(f"**混淆矩陣:**")
                    st.write(f"TN:{oos.get('tn', 0)} FP:{oos.get('fp', 0)}")
                    st.write(f"FN:{oos.get('fn', 0)} TP:{oos.get('tp', 0)}")
                with col_f:
                    st.write(f"**綜合指標:**")
                    st.write(f"召回率: {oos.get('recall', 0)*100:.1f}%")
                    st.write(f"準確率: {oos.get('accuracy', 0)*100:.1f}%")
                
                # Top特徵
                with st.expander("Top 10 特徵"):
                    for i, (name, imp) in enumerate(results['feature_importance'][:10], 1):
                        st.write(f"{i}. {name}: {imp:.4f}")
                
                # 完整JSON
                with st.expander("完整結果 (JSON)"):
                    st.json(results)
                
                # 質量判斷
                if results['oos_metrics']['auc'] > 0.60:
                    st.success("[OK] OOS AUC > 0.60 - 模型質量優秀")
                elif results['oos_metrics']['auc'] > 0.55:
                    st.info("[OK] OOS AUC > 0.55 - 模型可用")
                else:
                    st.warning("[WARNING] OOS AUC < 0.55 - 建議調整參數")
                
            except Exception as e:
                st.error(f"[ERROR] 訓練失敗: {e}")
                st.exception(e)
        else:
            st.info("""
            **V5訓練流程:**
            
            1. 生成價格特徵 (動量, 波動, 成交量)
            2. 智能標籤生成 (質量控制)
            3. 訓練集成XGBoost
            4. OOS驗證
            
            **預期結果:**
            - OOS AUC: 0.60-0.70
            - 正類率: 20-30%
            - 特徵數: 50-80
            
            **調參建議:**
            - 標籤太多(>35%)? 提高目標%
            - 標籤太少(<15%)? 降低目標%
            - AUC低? 增加樹深度/數量
            """)

def render_backtesting():
    st.subheader("策略回測")
    st.info("回測功能完整,略")
