"""
Smart Trading Terminal - Main GUI
STT 主界面

這個檔案僅負責:
1. 渲染界面結構
2. 呼叫版本模組
3. 管理頁面切換

所有逻輯由strategies/v*/模組處理
"""
import streamlit as st
import sys
from pathlib import Path

# 添加路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.version_manager import VersionManager
from core.data_loader import DataLoader
from core.gui_components import GUIComponents

# 頁面配置
st.set_page_config(
    page_title="Smart Trading Terminal",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """主函數"""
    
    # 標題
    st.title("🚀 Smart Trading Terminal")
    st.caption("模塊化加密貨幣交易策略系統")
    
    # 側邊欄 - 版本選擇
    with st.sidebar:
        st.header("📦 版本選擇")
        
        vm = VersionManager()
        versions = vm.list_versions()
        
        if not versions:
            st.error("⚠️ 沒有可用的策略版本")
            st.info("""
            請確保 strategies/ 資料夾中至少有一個版本
            例如: strategies/v1/
            """)
            return
        
        selected_version = st.selectbox(
            "選擇策略版本",
            versions,
            format_func=lambda x: x.upper()
        )
        
        # 顯示版本信息
        version_info = vm.get_version_info(selected_version)
        
        with st.expander("📊 版本信息"):
            st.write(f"**名稱**: {version_info['name']}")
            st.write(f"**訓練模組**: {'\u2705' if version_info['has_trainer'] else '\u274c'}")
            st.write(f"**回測模組**: {'\u2705' if version_info['has_backtester'] else '\u274c'}")
        
        st.markdown("---")
        
        # 數據信息
        st.header("📈 數據信息")
        st.info(f"""
        **數據源**: HuggingFace
        **交易對**: {len(DataLoader.SYMBOLS)}個
        **時間框架**: {len(DataLoader.TIMEFRAMES)}個
        """)
    
    # 主內容區 - 加載版本模組
    try:
        version_module = vm.get_version(selected_version)
        
        # 呼叫版本的render函數
        if hasattr(version_module, 'render'):
            version_module.render()
        else:
            st.error(f"版本 {selected_version} 沒有render()函數")
            st.info("""
            請確保 strategies/{version}/__init__.py 中定義了 render() 函數
            """)
    
    except Exception as e:
        st.error(f"加載版本失敗: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()
