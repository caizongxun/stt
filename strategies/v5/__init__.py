"""
V5 Strategy
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
    st.header("V5 - Dual Model System")
    tab1, tab2 = st.tabs(["訓練", "回測"])
    with tab1:
        render_training()
    with tab2:
        render_backtesting()

def render_training():
    col1, col2 = st.columns([1, 2])
    with col1:
        symbol, timeframe = GUIComponents.render_data_selector()
        forward_bars = st.slider("預測周期", 4, 16, 8, 2)
        min_return = st.slider("目標%", 0.5, 2.0, 0.8, 0.1)
        train_button = st.button("訓練", type="primary", use_container_width=True)
    with col2:
        if train_button:
            loader = DataLoader()
            df = loader.load_klines(symbol, timeframe)
            config = V5Config(symbol=symbol, timeframe=timeframe, forward_bars=forward_bars, min_return_pct=min_return/100)
            trainer = V5Trainer(config)
            results = trainer.train(df)
            st.json(results)

def render_backtesting():
    models_dir = Path('models')
    v5_models = [d for d in models_dir.iterdir() if d.is_dir() and '_v5_' in d.name] if models_dir.exists() else []
    if not v5_models:
        st.warning("無模型")
        return
    
    col1, col2 = st.columns([1, 2])
    with col1:
        model_names = [m.name for m in sorted(v5_models, reverse=True)]
        selected_model = st.selectbox("模型", model_names)
        backtest_days = st.slider("天數", 7, 90, 30)
        long_threshold = st.slider("LONG", 0.3, 0.9, 0.6, 0.05)
        short_threshold = st.slider("SHORT", 0.3, 0.9, 0.6, 0.05)
        leverage = st.slider("槓桶", 1, 10, 5)
        backtest_button = st.button("回測", type="primary", use_container_width=True)
    
    with col2:
        if backtest_button:
            model_path = models_dir / selected_model
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
            
            loader = DataLoader()
            df = loader.load_klines(config_dict['symbol'], config_dict['timeframe'])
            df = df.tail(backtest_days * 96)
            
            config = V5Config(symbol=config_dict['symbol'], timeframe=config_dict['timeframe'])
            config.long_threshold = long_threshold
            config.short_threshold = short_threshold
            config.leverage = leverage
            
            backtester = V5Backtester(config)
            results = backtester.run(long_models, short_models, df, feature_names)
            
            if results['status'] == 'success':
                ret = results['capital']['total_return_pct']
                monthly = results['capital']['monthly_return_pct']
                st.metric("總報酬", f"{ret:.1f}%")
                st.metric("月化", f"{monthly:.1f}%")
                st.json(results)
            else:
                st.warning("無交易")
