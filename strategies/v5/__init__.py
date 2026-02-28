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
    st.header("🚀 V5 Strategy - Pure ML")
    st.success("""
    **極簡高效系統**
    
    ✅ 不用市場狀態分類
    ✅ 不用支撑/壓力識別
    ✅ 不用任何硬規則過濾
    ✅ 完全信任 ML 預測
    
    🎯 **目標: 月報酬 20-50%**
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
        train_button = st.button("開始訓練 🚀", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 訓練過程")
        
        if train_button:
            with st.spinner("加載數據..."):
                loader = DataLoader()
                df = loader.load_klines(symbol, timeframe)
                st.success(f"✅ {len(df)} bars")
            
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
                
                st.success("✅ 訓練完成!")
                
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
                with st.expander("📈 Top 10 特徵"):
                    for i, (name, imp) in enumerate(results['feature_importance'][:10], 1):
                        st.write(f"{i}. **{name}**: {imp:.4f}")
                
                # 質量判斷
                if results['oos_metrics']['auc'] > 0.60:
                    st.success("✅ OOS AUC > 0.60 - 模型質量優秀!")
                elif results['oos_metrics']['auc'] > 0.55:
                    st.info("👍 OOS AUC > 0.55 - 模型可用")
                else:
                    st.warning("⚠️ OOS AUC < 0.55 - 建議調整參數")
                
            except Exception as e:
                st.error(f"❌ 訓練失敗: {e}")
                st.exception(e)
        else:
            st.info("""
            **V5訓練流程:**
            
            1️⃣ 生成價格特徵 (動量, 波動, 成交量)
            2️⃣ 智能標籤生成 (質量控制)
            3️⃣ 訓練集成XGBoost
            4️⃣ OOS驗證
            
            **預期結果:**
            - OOS AUC: 0.60-0.70
            - 正類率: 20-30%
            - 特徵數: 50-80
            
            **調參建議:**
            - 標籤太多(ャ5%)? 提高目標%
            - 標籤太少(<15%)? 降低目標%
            - AUC低? 增加樹深度/數量
            """)

def render_backtesting():
    st.subheader("策略回測")
    
    models_dir = Path('models')
    if not models_dir.exists():
        st.warning("沒有models資料夾")
        return
    
    v5_models = [d for d in models_dir.iterdir() if d.is_dir() and '_v5_' in d.name]
    if not v5_models:
        st.warning("沒有V5模型")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 回測參數")
        
        model_names = [m.name for m in sorted(v5_models, key=lambda x: x.name, reverse=True)]
        selected_model = st.selectbox("選擇模型", model_names)
        
        st.markdown("---")
        backtest_days = st.slider("回測天數", 7, 90, 30, 7)
        
        st.markdown("**交易參數**")
        predict_threshold = st.slider("預測閉值", 0.4, 0.8, 0.55, 0.05)
        
        st.markdown("**資金管理**")
        leverage = st.slider("槓桶倍數", 1, 10, 5, 1)
        position_pct = st.slider("仓位比例", 0.1, 0.6, 0.4, 0.05)
        use_compound = st.checkbox("複利模式", value=True)
        
        st.markdown("**風控參數**")
        atr_sl = st.slider("止損(ATR)", 1.0, 3.0, 2.0, 0.5)
        atr_tp = st.slider("止盈(ATR)", 2.0, 6.0, 4.0, 0.5)
        use_trailing = st.checkbox("移動止損", value=True)
        
        st.markdown("---")
        backtest_button = st.button("開始回測 📈", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 回測結果")
        
        if backtest_button:
            model_path = models_dir / selected_model
            
            try:
                with st.spinner("加載模型..."):
                    models = []
                    i = 0
                    while True:
                        model_file = model_path / f'model_{i}.pkl'
                        if not model_file.exists():
                            break
                        models.append(joblib.load(model_file))
                        i += 1
                    
                    config_dict = joblib.load(model_path / 'config.pkl')
                    feature_names = joblib.load(model_path / 'features.pkl')
                    st.success(f"✅ {len(models)}個模型")
                
                with st.spinner("加載數據..."):
                    loader = DataLoader()
                    df = loader.load_klines(config_dict['symbol'], config_dict['timeframe'])
                    df = df.tail(backtest_days * 96)
                    st.success(f"✅ {len(df)} bars")
                
                config = V5Config(**config_dict)
                config.predict_threshold = predict_threshold
                config.leverage = leverage
                config.position_pct = position_pct
                config.use_compound = use_compound
                config.atr_sl_multiplier = atr_sl
                config.atr_tp_multiplier = atr_tp
                config.use_trailing_stop = use_trailing
                
                backtester = V5Backtester(config)
                
                with st.spinner("執行回測..."):
                    results = backtester.run(models, df, feature_names)
                
                if results['status'] == 'no_trades':
                    st.warning("⚠️ 無交易信號 - 嘗試降低預測閉值")
                else:
                    st.success("✅ 回測完成!")
                    
                    # 核心指標
                    st.markdown("### 🎯 核心結果")
                    col_a, col_b, col_c, col_d = st.columns(4)
                    
                    ret = results['capital']['total_return_pct']
                    wr = results['trades']['win_rate_pct']
                    pf = results['trades']['profit_factor']
                    dd = results['capital']['max_drawdown_pct']
                    
                    with col_a:
                        delta_color = "normal" if ret > 0 else "inverse"
                        st.metric("總報酬", f"{ret:.1f}%", 
                                 delta="優" if ret > 30 else "中" if ret > 15 else "弱",
                                 delta_color=delta_color)
                    with col_b:
                        st.metric("勝率", f"{wr:.1f}%",
                                 delta="優" if wr > 55 else "中" if wr > 50 else "弱")
                    with col_c:
                        st.metric("利潤因子", f"{pf:.2f}",
                                 delta="優" if pf > 2 else "中" if pf > 1.5 else "弱")
                    with col_d:
                        st.metric("最大回撤", f"{dd:.1f}%",
                                 delta="優" if dd < 15 else "中" if dd < 25 else "危險",
                                 delta_color="inverse")
                    
                    # 月化報酬
                    days = results['period']['days']
                    monthly_return = ret * 30 / days if days > 0 else 0
                    
                    if monthly_return >= 20:
                        st.success(f"🎉 月化報酬: {monthly_return:.1f}% - 達成目標!")
                    elif monthly_return >= 10:
                        st.info(f"👍 月化報酬: {monthly_return:.1f}% - 接近目標")
                    else:
                        st.warning(f"⚠️ 月化報酬: {monthly_return:.1f}% - 未達目標")
                    
                    # 交易統計
                    st.markdown("### 📊 交易統計")
                    col_e, col_f, col_g, col_h = st.columns(4)
                    with col_e:
                        st.metric("總交易", results['trades']['total'])
                    with col_f:
                        st.metric("獲利", results['trades']['winning'])
                    with col_g:
                        st.metric("虧損", results['trades']['losing'])
                    with col_h:
                        avg_w = results['trades']['avg_win']
                        avg_l = abs(results['trades']['avg_loss'])
                        ratio = avg_w/avg_l if avg_l > 0 else 0
                        st.metric("盈虧比", f"{ratio:.2f}")
                    
                    # 出場統計
                    with st.expander("📝 出場原因"):
                        st.json(results['exit_reasons'])
                    
                    # 近期交易
                    with st.expander("📋 近期交易"):
                        for trade in results['trades_sample']:
                            pnl_emoji = "🟢" if trade['pnl'] > 0 else "🔴"
                            st.write(f"{pnl_emoji} {trade['direction']} | 盈虧: {trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%) | {trade['exit_reason']}")
            
            except Exception as e:
                st.error(f"❌ 回測失敗: {e}")
                st.exception(e)
        else:
            st.info("""
            **💡 回測說明:**
            
            **預測閉值:**
            - 0.45-0.50: 激進 (多交易)
            - 0.55: 平衡 (推薦)
            - 0.60-0.70: 保守 (少交易)
            
            **槓桶建議:**
            - 3-5x: 中等風險 (推薦)
            - 5-8x: 高風險 (激進)
            
            **移動止損:**
            - 啟動: 盈利 > 1.5%
            - 距離: 0.8%
            - 保護利潤,讓盈利奔跑
            """)
