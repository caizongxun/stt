"""
Reusable GUI Components
可重用的GUI組件
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Optional
from datetime import datetime

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
        渲柕績效指標
        
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
    
    @staticmethod
    def render_news_panel(
        news_list: List[Dict],
        show_filters: bool = True,
        max_display: int = 20
    ):
        """
        渲染新聞面板
        
        Args:
            news_list: 新聞列表
            show_filters: 是否顯示過濾器
            max_display: 最大顯示數量
        """
        if not news_list:
            st.info("📰 無新聞資料")
            return
        
        st.markdown("---")
        st.subheader("📰 最新市場新聞")
        
        # 過濾器
        if show_filters:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # 來源過濾
                all_sources = list(set(item['source'] for item in news_list))
                selected_sources = st.multiselect(
                    "來源過濾",
                    all_sources,
                    default=all_sources
                )
            
            with col2:
                # 分類過濾
                all_categories = list(set(item['category'] for item in news_list))
                selected_categories = st.multiselect(
                    "分類過濾",
                    all_categories,
                    default=all_categories
                )
            
            with col3:
                # 關鍵字搜尋
                keyword_filter = st.text_input(
                    "🔍 關鍵字搜尋",
                    placeholder="輸入關鍵字..."
                )
            
            # 應用過濾
            filtered_news = [
                item for item in news_list
                if item['source'] in selected_sources
                and item['category'] in selected_categories
            ]
            
            if keyword_filter:
                filtered_news = [
                    item for item in filtered_news
                    if keyword_filter.lower() in item['title'].lower()
                    or keyword_filter.lower() in item.get('summary', '').lower()
                ]
        else:
            filtered_news = news_list
        
        # 顯示統計
        col1, col2, col3 = st.columns(3)
        col1.metric("總數", len(filtered_news))
        col2.metric("來源數", len(set(item['source'] for item in filtered_news)))
        
        if filtered_news:
            latest_time = max(item['published'] for item in filtered_news)
            time_ago = (datetime.now() - latest_time).total_seconds() / 60
            col3.metric("最新", f"{int(time_ago)} 分鐘前")
        
        st.markdown("---")
        
        # 顯示新聞
        for idx, item in enumerate(filtered_news[:max_display], 1):
            with st.expander(
                f"📌 {item['title']}",
                expanded=(idx == 1)  # 預設展開第一條
            ):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # 內容
                    if item.get('full_content'):
                        st.markdown(f"**完整內容**:")
                        st.write(item['full_content'][:500] + "...")
                    else:
                        st.markdown(f"**摘要**:")
                        st.write(item.get('summary', '無摘要'))
                    
                    # 鏈接
                    st.markdown(f"[🔗 閱讀全文]({item['link']})")
                
                with col2:
                    # 元資料
                    st.markdown(f"""
                    **來源**: {item['source']}  
                    **分類**: {item['category']}  
                    **時間**: {item['published'].strftime('%m-%d %H:%M')}
                    """)
                    
                    # 圖片
                    if item.get('image_url'):
                        try:
                            st.image(item['image_url'], width=200)
                        except:
                            pass
        
        # 分頁提示
        if len(filtered_news) > max_display:
            st.info(f"ℹ️ 顯示前 {max_display} 條,共 {len(filtered_news)} 條新聞")
    
    @staticmethod
    def render_news_statistics(news_list: List[Dict]):
        """
        渲染新聞統計
        
        Args:
            news_list: 新聞列表
        """
        if not news_list:
            return
        
        st.subheader("📊 新聞統計")
        
        # 按來源統計
        source_counts = {}
        category_counts = {}
        
        for item in news_list:
            source = item['source']
            category = item['category']
            
            source_counts[source] = source_counts.get(source, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**按來源**")
            for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
                st.write(f"- {source}: {count} 條")
        
        with col2:
            st.markdown("**按分類**")
            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                st.write(f"- {category}: {count} 條")
