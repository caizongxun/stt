"""
V6 Strategy - Reversal Prediction
"""
import joblib
import streamlit as st
from pathlib import Path

from core.gui_components import GUIComponents
from core.data_loader import DataLoader
from .config import V6Config
from .trainer import V6Trainer
from .backtester import V6Backtester


def render():
    st.header("V6 - 反轉預測模型")
    st.info(
        """
        **目標:** 捕捉極端位置的反轉機會  
        - 使用BB位置 + RSI 監測超買/超賣  
        - LightGBM 估計反轉成功率  
        - 內建風控: ATR 止損 / 止盈、持倉上限
        """
    )

    tab1, tab2 = st.tabs(["訓練", "回測"])
    with tab1:
        render_training()
    with tab2:
        render_backtesting()


def render_training():
    col1, col2 = st.columns([1, 2])
    with col1:
        symbol, timeframe = GUIComponents.render_data_selector()
        forward_bars = st.slider("預測持續K數", 4, 12, 6, 1)
        target_pct = st.slider("目標反轉幅度(%)", 0.3, 1.5, 0.6, 0.1)
        oversold = st.slider("超賣RSI", 20, 40, 32, 1)
        overbought = st.slider("超買RSI", 60, 80, 68, 1)
        train_btn = st.button("開始訓練", type="primary", use_container_width=True)

    with col2:
        if train_btn:
            with st.spinner("加載數據..."):
                loader = DataLoader()
                df = loader.load_klines(symbol, timeframe)
                if df.empty:
                    st.error("無數據")
                    return
                st.success(f"取得 {len(df)} 筆")

            config = V6Config(
                symbol=symbol,
                timeframe=timeframe,
                forward_bars=forward_bars,
                reversal_target_pct=target_pct / 100,
                oversold_rsi=oversold,
                overbought_rsi=overbought,
            )

            trainer = V6Trainer(config)
            with st.spinner("訓練模型..."):
                results = trainer.train(df)

            st.success("訓練完成")
            st.json(results)
        else:
            st.info("調整參數後點擊「開始訓練」")


def render_backtesting():
    models_dir = Path("models")
    v6_models = (
        [d for d in models_dir.iterdir() if d.is_dir() and "_v6_" in d.name]
        if models_dir.exists()
        else []
    )

    if not v6_models:
        st.warning("尚無V6模型，請先訓練")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        model_names = [m.name for m in sorted(v6_models, reverse=True)]
        selected = st.selectbox("選擇模型", model_names)
        backtest_days = st.slider("回測天數", 14, 120, 45, 7)
        long_th = st.slider("Long門檻", 0.4, 0.8, 0.55, 0.05)
        short_th = st.slider("Short門檻", 0.4, 0.8, 0.55, 0.05)
        leverage = st.slider("槓桿", 1, 5, 2, 1)
        position_pct = st.slider("單筆倉位比例", 0.05, 0.4, 0.25, 0.05)
        atr_sl = st.slider("止損(ATR倍數)", 1.0, 3.0, 1.6, 0.1)
        atr_tp = st.slider("止盈(ATR倍數)", 1.5, 4.0, 2.8, 0.1)
        backtest_btn = st.button("開始回測", type="primary", use_container_width=True)

    with col2:
        if backtest_btn:
            model_path = models_dir / selected
            try:
                long_model = joblib.load(model_path / "long_model.pkl")
                short_model = joblib.load(model_path / "short_model.pkl")
                config_dict = joblib.load(model_path / "config.pkl")
                feature_names = joblib.load(model_path / "features.pkl")
            except Exception as e:
                st.error(f"模型載入失敗: {e}")
                return

            loader = DataLoader()
            df = loader.load_klines(config_dict["symbol"], config_dict["timeframe"])
            df = df.tail(backtest_days * 96)

            config = V6Config(**config_dict)
            config.long_threshold = long_th
            config.short_threshold = short_th
            config.leverage = leverage
            config.position_pct = position_pct
            config.atr_sl_multiplier = atr_sl
            config.atr_tp_multiplier = atr_tp

            backtester = V6Backtester(config)
            with st.spinner("執行回測..."):
                results = backtester.run(long_model, short_model, df, feature_names)

            if results["status"] == "success":
                st.metric("總報酬率", f"{results['capital']['total_return_pct']:.1f}%")
                st.metric("勝率", f"{results['trades']['win_rate_pct']:.1f}%")
                st.metric("利潤因子", f"{results['trades']['profit_factor']:.2f}")
                st.json(results)
            else:
                st.warning("未產生交易，調整門檻或天數再試")
        else:
            st.info("選擇模型後點擊回測")
