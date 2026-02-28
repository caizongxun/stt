"""
Streamlit Trading Tool
交易系統主程式
"""
import streamlit as st
from strategies import v1, v2, v3

# 頁面配置
st.set_page_config(
    page_title="Trading System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 標題
st.title("📊 AI Trading System")

# 側邊欄選擇策略
st.sidebar.title("策略選擇")
strategy = st.sidebar.radio(
    "選擇交易策略",
    [
        "V1 - 基礎版",
        "V2 - BB反轉",
        "V3 - 高性能 (50%/30天)"
    ],
    index=2
)

# 策略說明
st.sidebar.markdown("---")
if "V1" in strategy:
    st.sidebar.info("""
    **V1 基礎版**
    - 單一策略
    - 無OOS驗證
    - 適合學習
    """)
elif "V2" in strategy:
    st.sidebar.info("""
    **V2 BB反轉**
    - BB通道反轉
    - ATR風控
    - 適合穩健交易
    """)
else:
    st.sidebar.warning("""
    **V3 高性能**
    - 多策略融合
    - 嚴格OOS驗證
    - 5倍槓桶
    - 目標50%/30天
    - ⚠️ 高風險
    """)

# 渲染對應策略
if "V1" in strategy:
    v1.render()
elif "V2" in strategy:
    v2.render()
else:
    v3.render()
