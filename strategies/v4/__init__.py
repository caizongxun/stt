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
    
    tab1, tab2, tab3 = st.tabs(["訓練", "回測", "參數優化"])
    
    with tab1:
        render_training()
    
    with tab2:
        render_backtesting()
    
    with tab3:
        render_optimization()

def render_training():
    """(略)"""
    pass

def render_backtesting():
    st.subheader("策略回測")
    
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
        st.markdown("### 回測參數")
        
        model_names = [m.name for m in sorted(v4_models, key=lambda x: x.name, reverse=True)]
        selected_model = st.selectbox("選擇模型", model_names)
        
        st.markdown("---")
        backtest_days = st.slider("回測天數", 7, 90, 30, 7)
        
        st.markdown("**信號模式**")
        signal_mode = st.radio(
            "選擇信號模式",
            ['pure', 'hybrid', 'ranging', 'trending'],
            format_func=lambda x: {
                'pure': 'Pure - 完全信任模型',
                'hybrid': 'Hybrid - 混合模式',
                'ranging': 'Ranging - 僅盤整',
                'trending': 'Trending - 僅趨勢'
            }[x],
            index=0
        )
        
        st.markdown("**交易參數**")
        predict_threshold = st.slider("預測閉值", 0.3, 0.9, 0.5, 0.05)
        
        st.markdown("**資金管理**")
        leverage = st.slider("槓桶倍數", 1, 10, 3, 1)
        position_pct = st.slider("仓位比例", 0.1, 0.5, 0.3, 0.05)
        use_compound = st.checkbox("複利模式", value=True)
        
        st.markdown("**風控參數**")
        atr_sl = st.slider("止損(ATR倍數)", 0.5, 3.0, 1.5, 0.5)
        atr_tp_range = st.slider("盤整止盈", 1.0, 3.0, 1.5, 0.5)
        atr_tp_breakout = st.slider("趨勢止盈", 2.0, 5.0, 3.0, 0.5)
        
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
                
                # 使用自定義參數
                config = V4Config(**config_dict)
                config.signal_mode = signal_mode
                config.predict_threshold = predict_threshold
                config.leverage = leverage
                config.position_pct = position_pct
                config.use_compound = use_compound
                config.atr_sl_multiplier = atr_sl
                config.atr_tp_range = atr_tp_range
                config.atr_tp_breakout = atr_tp_breakout
                
                backtester = V4Backtester(config)
                
                with st.spinner("執行回測..."):
                    results = backtester.run(models, df, feature_names)
                
                if results['status'] == 'no_trades':
                    st.warning("無交易信號")
                    if 'debug' in results:
                        st.json(results['debug'])
                else:
                    st.success("[OK] 回測完成")
                    
                    # 核心指標
                    st.markdown("### 核心結果")
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        ret = results['capital']['total_return_pct']
                        st.metric("總報酬", f"{ret:.1f}%")
                    with col_b:
                        wr = results['trades']['win_rate_pct']
                        st.metric("勝率", f"{wr:.1f}%")
                    with col_c:
                        pf = results['trades']['profit_factor']
                        st.metric("利潤因子", f"{pf:.2f}")
                    with col_d:
                        dd = results['capital']['max_drawdown_pct']
                        st.metric("最大回撤", f"{dd:.1f}%")
                    
                    # 交易統計
                    st.markdown("### 交易統計")
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
                        st.metric("平均盈虧比", f"{avg_w/avg_l if avg_l > 0 else 0:.2f}")
                    
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
                    
                    with st.expander("完整JSON"):
                        st.json(results)
            
            except Exception as e:
                st.error(f"回測失敗: {e}")
                st.exception(e)
        else:
            st.info("""
            **信號模式說明:**
            
            - **Pure**: 完全信任模型預測,用價格動量判斷方向
            - **Hybrid**: 混合使用原始信號+模型預測
            - **Ranging**: 僅使用盤整信號(支撑/壓力)
            - **Trending**: 僅使用趨勢信號(突破)
            
            **推薦從 Pure 模式開始測試!**
            """)

def render_optimization():
    """(略)"""
    pass
