"""
V5 Strategy - Dual Model System
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
        render_backtesting()

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
            
            config = V5Config(symbol=symbol, timeframe=timeframe, forward_bars=forward_bars,
                            min_return_pct=min_return/100, require_no_reverse=require_no_reverse,
                            max_depth=max_depth, n_estimators=n_estimators, ensemble_models=ensemble_models)
            trainer = V5Trainer(config)
            
            try:
                with st.spinner("訓練中..."):
                    results = trainer.train(df)
                st.success("[OK] 雙模型訓練完成")
                
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
                    st.metric("閉值", f"{long_oos.get('threshold', 0.5):.2f}")
                
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
                    st.metric("閉值", f"{short_oos.get('threshold', 0.5):.2f}")
                
                col_i, col_j = st.columns(2)
                with col_i:
                    with st.expander("LONG Top Features"):
                        for i, (name, imp) in enumerate(results['long_feature_importance'], 1):
                            st.write(f"{i}. {name}: {imp:.4f}")
                with col_j:
                    with st.expander("SHORT Top Features"):
                        for i, (name, imp) in enumerate(results['short_feature_importance'], 1):
                            st.write(f"{i}. {name}: {imp:.4f}")
                
                with st.expander("完整結果 (JSON)"):
                    st.json(results)
                
                avg_auc = (long_oos['auc'] + short_oos['auc']) / 2
                if avg_auc > 0.60:
                    st.success(f"[OK] 平均AUC {avg_auc:.3f}")
                elif avg_auc > 0.55:
                    st.info(f"[OK] 平均AUC {avg_auc:.3f}")
                else:
                    st.warning(f"[WARNING] 平均AUC {avg_auc:.3f}")
            except Exception as e:
                st.error(f"[ERROR] {e}")
                st.exception(e)
        else:
            st.info("選擇參數並開始訓練")

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
        
        st.markdown("**預測閉值**")
        long_threshold = st.slider("LONG閉值", 0.3, 0.9, 0.6, 0.05)
        short_threshold = st.slider("SHORT閉值", 0.3, 0.9, 0.6, 0.05)
        
        st.markdown("**資金管理**")
        leverage = st.slider("槓桶", 1, 10, 5, 1)
        position_pct = st.slider("仓位%", 0.1, 0.6, 0.4, 0.05)
        use_compound = st.checkbox("複利", value=True)
        
        st.markdown("**風控**")
        atr_sl = st.slider("止損(ATR)", 1.0, 3.0, 2.0, 0.5)
        atr_tp = st.slider("止盈(ATR)", 2.0, 6.0, 4.0, 0.5)
        use_trailing = st.checkbox("移動止損", value=True)
        
        st.markdown("---")
        backtest_button = st.button("開始回測", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 回測結果")
        
        if backtest_button:
            model_path = models_dir / selected_model
            
            try:
                with st.spinner("加載模型..."):
                    long_models, short_models = [], []
                    i = 0
                    while (model_path / f'long_model_{i}.pkl').exists():
                        long_models.append(joblib.load(model_path / f'long_model_{i}.pkl'))
                        i += 1
                    i = 0
                    while (model_path / f'short_model_{i}.pkl').exists():
                        short_models.append(joblib.load(model_path / f'short_model_{i}.pkl'))
                        i += 1
                    config_dict = joblib.load(model_path / 'config.pkl')
                    feature_names = joblib.load(model_path / 'features.pkl')
                    st.success(f"[OK] LONG:{len(long_models)} SHORT:{len(short_models)}")
                
                with st.spinner("加載數據..."):
                    loader = DataLoader()
                    df = loader.load_klines(config_dict['symbol'], config_dict['timeframe'])
                    df = df.tail(backtest_days * 96)
                    st.success(f"[OK] {len(df)} bars")
                
                config = V5Config(**config_dict)
                config.long_threshold = long_threshold
                config.short_threshold = short_threshold
                config.leverage = leverage
                config.position_pct = position_pct
                config.use_compound = use_compound
                config.atr_sl_multiplier = atr_sl
                config.atr_tp_multiplier = atr_tp
                config.use_trailing_stop = use_trailing
                
                backtester = V5Backtester(config)
                
                with st.spinner("執行回測..."):
                    results = backtester.run(long_models, short_models, df, feature_names)
                
                if results['status'] == 'no_trades':
                    st.warning("[WARNING] 無交易")
                else:
                    st.success("[OK] 回測完成")
                    
                    st.markdown("### 核心結果")
                    col_a, col_b, col_c, col_d = st.columns(4)
                    ret = results['capital']['total_return_pct']
                    monthly = results['capital']['monthly_return_pct']
                    wr = results['trades']['win_rate_pct']
                    pf = results['trades']['profit_factor']
                    dd = results['capital']['max_drawdown_pct']
                    
                    with col_a:
                        st.metric("總報酬", f"{ret:.1f}%")
                    with col_b:
                        st.metric("月化", f"{monthly:.1f}%")
                    with col_c:
                        st.metric("勝率", f"{wr:.1f}%")
                    with col_d:
                        st.metric("回撤", f"{dd:.1f}%")
                    
                    if monthly >= 20:
                        st.success(f"[SUCCESS] 月化{monthly:.1f}% - 達成目標!")
                    elif monthly >= 10:
                        st.info(f"[GOOD] 月化{monthly:.1f}%")
                    else:
                        st.warning(f"[LOW] 月化{monthly:.1f}%")
                    
                    st.markdown("### 交易統計")
                    col_e, col_f, col_g, col_h = st.columns(4)
                    with col_e:
                        st.metric("總交易", results['trades']['total'])
                    with col_f:
                        st.metric("LONG", f"{results['trades']['long_trades']} ({results['trades']['long_win_rate']:.0f}%)")
                    with col_g:
                        st.metric("SHORT", f"{results['trades']['short_trades']} ({results['trades']['short_win_rate']:.0f}%)")
                    with col_h:
                        st.metric("利潤因子", f"{pf:.2f}")
                    
                    with st.expander("出場統計"):
                        st.json(results['exit_reasons'])
                    
                    with st.expander("近期交易"):
                        for trade in results['trades_sample']:
                            emoji = "[+]" if trade['pnl'] > 0 else "[-]"
                            st.write(f"{emoji} {trade['direction']} | PnL:{trade['pnl']:.1f} ({trade['pnl_pct']:.2f}%) | {trade['exit_reason']}")
                    
                    with st.expander("完整結果 (JSON)"):
                        st.json(results)
            
            except Exception as e:
                st.error(f"[ERROR] {e}")
                st.exception(e)
        else:
            st.info("選擇模型並設定參數")
