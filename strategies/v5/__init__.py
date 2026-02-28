"""
V5 Strategy - Dual Model System
V5策略 - 雙模型系統
"""
import streamlit as st
from pathlib import Path
from core.gui_components import GUIComponents
from core.data_loader import DataLoader
from .config import V5Config
from .trainer import V5Trainer

def render():
    st.header("V5 Strategy - Dual Model System")
    st.success("""
    **雙模型系統 + 增強特徵**
    
    [NEW] 做多/做空分離訓練
    [NEW] K線型態識別
    [NEW] 支撓/壓力距離
    [NEW] 成交量突變檢測
    
    [TARGET] 月報酬 20-50%
    """)
    
    tab1, tab2 = st.tabs(["訓練", "回測"])
    
    with tab1:
        render_training()
    
    with tab2:
        st.info("回測功能開發中...")

def render_training():
    st.subheader("模型訓練")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 訓練參數")
        
        symbol, timeframe = GUIComponents.render_data_selector()
        
        st.markdown("---")
        st.markdown("**標籤參數**")
        forward_bars = st.slider("預測周期", 4, 16, 8, 2)
        min_return = st.slider("最小目標%", 0.5, 2.0, 0.8, 0.1)
        require_no_reverse = st.checkbox("要求不回撤", value=True)
        
        st.markdown("**模型參數**")
        max_depth = st.slider("樹深度", 4, 8, 6, 1)
        n_estimators = st.slider("樹數量", 100, 500, 300, 50)
        ensemble_models = st.slider("集成模型數", 3, 7, 5, 1)
        
        st.markdown("---")
        train_button = st.button("開始訓練", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 訓練結果")
        
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
                
                st.success("[OK] 雙模型訓練完成")
                
                # LONG模型指標
                st.markdown("### LONG Model")
                col_a, col_b, col_c, col_d = st.columns(4)
                long_oos = results['long_oos_metrics']
                with col_a:
                    st.metric("OOS AUC", f"{long_oos['auc']:.3f}")
                with col_b:
                    st.metric("精確率", f"{long_oos['precision']*100:.1f}%")
                with col_c:
                    st.metric("召回率", f"{long_oos['recall']*100:.1f}%")
                with col_d:
                    st.metric("準確率", f"{long_oos['accuracy']*100:.1f}%")
                
                # SHORT模型指標
                st.markdown("### SHORT Model")
                col_e, col_f, col_g, col_h = st.columns(4)
                short_oos = results['short_oos_metrics']
                with col_e:
                    st.metric("OOS AUC", f"{short_oos['auc']:.3f}")
                with col_f:
                    st.metric("精確率", f"{short_oos['precision']*100:.1f}%")
                with col_g:
                    st.metric("召回率", f"{short_oos['recall']*100:.1f}%")
                with col_h:
                    st.metric("準確率", f"{short_oos['accuracy']*100:.1f}%")
                
                # 混淆矩陣
                col_i, col_j = st.columns(2)
                with col_i:
                    st.write("**LONG 混淆矩陣:**")
                    st.write(f"TN:{long_oos.get('tn', 0)} FP:{long_oos.get('fp', 0)}")
                    st.write(f"FN:{long_oos.get('fn', 0)} TP:{long_oos.get('tp', 0)}")
                with col_j:
                    st.write("**SHORT 混淆矩陣:**")
                    st.write(f"TN:{short_oos.get('tn', 0)} FP:{short_oos.get('fp', 0)}")
                    st.write(f"FN:{short_oos.get('fn', 0)} TP:{short_oos.get('tp', 0)}")
                
                # Top特徵
                col_k, col_l = st.columns(2)
                with col_k:
                    with st.expander("LONG Top Features"):
                        for i, (name, imp) in enumerate(results['long_feature_importance'], 1):
                            st.write(f"{i}. {name}: {imp:.4f}")
                with col_l:
                    with st.expander("SHORT Top Features"):
                        for i, (name, imp) in enumerate(results['short_feature_importance'], 1):
                            st.write(f"{i}. {name}: {imp:.4f}")
                
                # 完整JSON
                with st.expander("完整結果 (JSON)"):
                    st.json(results)
                
                # 質量判斷
                avg_auc = (long_oos['auc'] + short_oos['auc']) / 2
                if avg_auc > 0.60:
                    st.success(f"[OK] 平均AUC {avg_auc:.3f} - 模型優秀")
                elif avg_auc > 0.55:
                    st.info(f"[OK] 平均AUC {avg_auc:.3f} - 模型可用")
                else:
                    st.warning(f"[WARNING] 平均AUC {avg_auc:.3f} - 建議調整")
                
            except Exception as e:
                st.error(f"[ERROR] 訓練失敗: {e}")
                st.exception(e)
        else:
            st.info("""
            **雙模型系統:**
            
            1. LONG Model - 專門預測做多機會
            2. SHORT Model - 專門預測做空機會
            3. 各自優化,不互相干擾
            
            **增強特徵:**
            - K線型態: 十字星/錐子/包容
            - 支撓壓力: 多時間框架距離
            - 成交量: 突增檢測/價量配合
            
            **預期改善:**
            - 召回率: 15% -> 30-40%
            - AUC: 0.66 -> 0.65-0.70
            - 特徵數: 49 -> 80-100
            """)
