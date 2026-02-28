"""
Reusable GUI Components
可重用的GUI組件
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

class GUIComponents:
    """
    共用GUI組件
    """
    
    @staticmethod
    def render_data_selector():
        """
        渲染數據選擇器
        
        Returns:
            tuple: (symbol, timeframe)
        """
        from core.data_loader import DataLoader
        
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.selectbox(
                "交易對",
                DataLoader.SYMBOLS,
                index=DataLoader.SYMBOLS.index('BTCUSDT')
            )
        
        with col2:
            timeframe = st.selectbox(
                "時間框架",
                DataLoader.TIMEFRAMES,
                index=DataLoader.TIMEFRAMES.index('15m')
            )
        
        return symbol, timeframe
    
    @staticmethod
    def render_training_params():
        """
        渲染訓練參數設置
        
        Returns:
            dict: 訓練參數
        """
        st.subheader("訓練參數")
        
        col1, col2 = st.columns(2)
        
        with col1:
            train_size = st.slider(
                "訓練集比例",
                0.5, 0.9, 0.7, 0.05
            )
            epochs = st.slider(
                "訓練輪數",
                10, 200, 50, 10
            )
        
        with col2:
            batch_size = st.select_slider(
                "批次大小",
                [16, 32, 64, 128, 256],
                value=64
            )
            learning_rate = st.select_slider(
                "學習率",
                [0.0001, 0.001, 0.01, 0.1],
                value=0.001,
                format_func=lambda x: f"{x:.4f}"
            )
        
        return {
            'train_size': train_size,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
    
    @staticmethod
    def render_backtest_params():
        """
        渲染回測參數設置
        
        Returns:
            dict: 回測參數
        """
        st.subheader("回測參數")
        
        col1, col2 = st.columns(2)
        
        with col1:
            capital = st.number_input(
                "初始資金 (USDT)",
                1000, 1000000, 10000, 1000
            )
            backtest_days = st.slider(
                "回測天數",
                30, 365, 90, 30
            )
        
        with col2:
            leverage = st.slider(
                "槓桿倍數",
                1, 10, 1, 1
            )
            fee_rate = st.number_input(
                "手續費率 (%)",
                0.0, 0.5, 0.1, 0.01
            )
        
        return {
            'capital': capital,
            'backtest_days': backtest_days,
            'leverage': leverage,
            'fee_rate': fee_rate / 100
        }
    
    @staticmethod
    def render_equity_curve(equity_data: pd.DataFrame):
        """
        渲染權益曲線
        
        Args:
            equity_data: 權益數據,包含timestamp和equity列
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=equity_data['timestamp'],
            y=equity_data['equity'],
            mode='lines',
            name='權益',
            line=dict(color='#00d4ff', width=2)
        ))
        
        fig.update_layout(
            title='權益曲線',
            xaxis_title='時間',
            yaxis_title='權益 (USDT)',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_metrics(metrics: dict):
        """
        渲染績效指標
        
        Args:
            metrics: 績效指標字典
        """
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "總報酬率",
                f"{metrics.get('total_return', 0)*100:.2f}%"
            )
            st.metric(
                "勝率",
                f"{metrics.get('win_rate', 0)*100:.2f}%"
            )
        
        with col2:
            st.metric(
                "總交易數",
                metrics.get('total_trades', 0)
            )
            st.metric(
                "盈虧比",
                f"{metrics.get('profit_factor', 0):.2f}"
            )
        
        with col3:
            st.metric(
                "Sharpe比率",
                f"{metrics.get('sharpe_ratio', 0):.2f}"
            )
            st.metric(
                "最大回撤",
                f"{metrics.get('max_drawdown', 0)*100:.2f}%"
            )
        
        with col4:
            st.metric(
                "平均盈利",
                f"${metrics.get('avg_win', 0):.2f}"
            )
            st.metric(
                "平均虧損",
                f"${metrics.get('avg_loss', 0):.2f}"
            )
