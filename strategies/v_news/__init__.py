"""
v_news - 新聞與 AI 分析示範

功能:
1. 抓取最新加密貨幣新聞
2. 建構給 AI 的 prompt
3. 顯示新聞列表
4. 支援關鍵字過濾
"""

import streamlit as st
from core.news_aggregator import NewsAggregator
from core.ai_prompt_builder import AIPromptBuilder
from core.gui_components import GUIComponents

# 版本資訊
VERSION_INFO = {
    'name': '新聞與 AI 分析',
    'description': '整合最新市場新聞與 AI 分析系統',
    'version': '0.1.0'
}

def render():
    """渲染主頁面"""
    
    st.header("📰 新聞與 AI 分析")
    st.caption("整合最新市場動態與 AI 分析")
    
    # Tab 切換
    tab1, tab2, tab3 = st.tabs([
        "📰 新聞瀏覽",
        "🤖 AI Prompt 建構",
        "⚙️ 設定"
    ])
    
    # 初始化新聞聚合器
    if 'news_aggregator' not in st.session_state:
        st.session_state.news_aggregator = NewsAggregator()
    
    aggregator = st.session_state.news_aggregator
    
    # Tab 1: 新聞瀏覽
    with tab1:
        render_news_browser(aggregator)
    
    # Tab 2: AI Prompt
    with tab2:
        render_ai_prompt(aggregator)
    
    # Tab 3: 設定
    with tab3:
        render_settings()

def render_news_browser(aggregator: NewsAggregator):
    """渲染新聞瀏覽器"""
    
    st.subheader("🔍 抓取最新新聞")
    
    # 控制面板
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        hours = st.slider("時間範圍 (小時)", 1, 72, 24)
    
    with col2:
        categories = st.multiselect(
            "分類",
            ['crypto', 'finance', 'taiwan'],
            default=['crypto']
        )
    
    with col3:
        include_content = st.checkbox(
            "爬取完整內容",
            value=False,
            help="會比較慢,但可獲得完整文章"
        )
    
    with col4:
        st.write("")
        st.write("")
        fetch_button = st.button("🔄 抓取新聞", use_container_width=True)
    
    # 抓取新聞
    if fetch_button or 'news_data' not in st.session_state:
        with st.spinner("正在抓取新聞..."):
            news_data = aggregator.fetch_all_news(
                hours=hours,
                include_content=include_content,
                categories=categories if categories else None
            )
            st.session_state.news_data = news_data
            st.success(f"✅ 成功抓取 {len(news_data)} 條新聞")
    
    # 顯示新聞
    if 'news_data' in st.session_state and st.session_state.news_data:
        news_data = st.session_state.news_data
        
        # 統計
        with st.expander("📊 新聞統計", expanded=False):
            GUIComponents.render_news_statistics(news_data)
        
        # 新聞面板
        GUIComponents.render_news_panel(
            news_data,
            show_filters=True,
            max_display=20
        )
    else:
        st.info("👆 點擊上方按鈕抓取最新新聞")

def render_ai_prompt(aggregator: NewsAggregator):
    """渲染 AI Prompt 建構器"""
    
    st.subheader("🤖 AI Prompt 建構器")
    
    if 'news_data' not in st.session_state or not st.session_state.news_data:
        st.warning("⚠️ 請先在「新聞瀏覽」頁面抓取新聞")
        return
    
    news_data = st.session_state.news_data
    builder = AIPromptBuilder()
    
    st.markdown("---")
    
    # 設定
    col1, col2 = st.columns(2)
    
    with col1:
        max_items = st.slider(
            "包含新聞數量",
            5, 30, 15
        )
    
    with col2:
        max_length = st.slider(
            "每條內容長度",
            500, 3000, 1500, 100
        )
    
    # 關鍵字過濾
    st.markdown("### 🔍 關鍵字過濾 (可選)")
    keywords_input = st.text_input(
        "輸入關鍵字 (逗號分隔)",
        placeholder="例: bitcoin,ethereum,btc,eth"
    )
    
    filtered_news = news_data
    if keywords_input:
        keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]
        filtered_news = aggregator.filter_by_keywords(news_data, keywords)
        st.info(f"🎯 過濾後: {len(filtered_news)} 條新聞")
    
    st.markdown("---")
    
    # 用戶問題
    st.markdown("### 💬 用戶問題")
    user_query = st.text_area(
        "輸入想問 AI 的問題",
        value="比特幣最近的走勢如何?有什麼重大新聞影響?",
        height=100
    )
    
    # 建構 Prompt
    if st.button("🚀 建構 Prompt", use_container_width=True):
        with st.spinner("正在建構..."):
            # 建構新聞 context
            news_context = builder.build_news_context(
                filtered_news,
                max_items=max_items,
                max_content_length=max_length,
                include_full_content=True
            )
            
            # 建構完整 prompt
            full_prompt = builder.build_complete_prompt(
                user_query=user_query,
                news_context=news_context
            )
            
            st.session_state.generated_prompt = full_prompt
            st.success("✅ Prompt 已生成!")
    
    # 顯示生成的 Prompt
    if 'generated_prompt' in st.session_state:
        st.markdown("---")
        st.markdown("### 📝 生成的 Prompt")
        
        prompt_length = len(st.session_state.generated_prompt)
        st.info(f"📏 Prompt 長度: {prompt_length:,} 字元 (~{prompt_length//4:,} tokens)")
        
        # 顯示 prompt
        with st.expander("👀 查看完整 Prompt", expanded=False):
            st.code(st.session_state.generated_prompt, language='markdown')
        
        # 複製按鈕
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 複製到剪貼板", use_container_width=True):
                st.code(st.session_state.generated_prompt)
                st.success("✅ 請手動複製上方內容")
        
        with col2:
            if st.button("💾 下載為文字檔", use_container_width=True):
                st.download_button(
                    label="⬇️ 下載 prompt.txt",
                    data=st.session_state.generated_prompt,
                    file_name="ai_prompt.txt",
                    mime="text/plain"
                )
        
        # 使用說明
        st.markdown("---")
        st.info("""
        💡 **使用方法**:
        1. 複製上方生成的 Prompt
        2. 貼到 ChatGPT / Claude / DeepSeek 等 AI 工具
        3. AI 將根據最新新聞回答你的問題
        """)

def render_settings():
    """渲染設定頁面"""
    
    st.subheader("⚙️ 新聞設定")
    
    st.markdown("""
    ### 📡 新聞來源
    
    當前支援的新聞來源:
    
    **加密貨幣**:
    - CoinDesk
    - CryptoSlate
    - Cointelegraph
    - Bitcoin Magazine
    - The Block
    
    **金融市場**:
    - Reuters Markets
    
    **台灣財經**:
    - 鉅亨網
    
    ---
    
    ### 🛠️ 技術說明
    
    **爬蟲方式**:
    - 使用 `requests` 庫,不開啟瀏覽器視窗
    - RSS feed 解析
    - 智能內容提取
    
    **快取機制**:
    - 記憶體快取: 30分鐘
    - 磁碟快取: 自動存檔
    
    **優勢**:
    - 不需要 Playwright/Selenium
    - 不開啟瀏覽器視窗
    - 資源使用低
    - 速度快
    
    ---
    
    ### 📚 使用教學
    
    1. **抓取新聞**: 在「新聞瀏覽」頁面點擊抓取
    2. **過濾內容**: 使用來源/分類/關鍵字過濾
    3. **建構 Prompt**: 在「AI Prompt」頁面生成
    4. **使用 AI**: 複製 prompt 到 AI 工具
    
    ---
    
    ### 🔗 相關連結
    
    - [WorldMonitor 原始專案](https://github.com/koala73/worldmonitor)
    - [Feedparser 文檔](https://feedparser.readthedocs.io/)
    - [BeautifulSoup 文檔](https://www.crummy.com/software/BeautifulSoup/)
    """)


if __name__ == "__main__":
    render()
