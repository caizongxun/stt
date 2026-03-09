# 🌍 WorldMonitor 新聞整合指南

## 簡介

這是一個從 [WorldMonitor](https://github.com/koala73/worldmonitor) 項目整合而來的全球新聞與市場動態系統。

**主要功能**:
- ✅ RSS Feed 自動爬取 (支援 RSS/Atom)
- ✅ 多來源新聞聚合 (CoinDesk, CryptoSlate, Reuters 等)
- ✅ 完整文章內容爬取 (不開啟瀏覽器)
- ✅ 自動威脅等級分類 (Critical/High/Medium/Low)
- ✅ 關鍵字過濾與搜尋
- ✅ 快取機制 (記憶體 + 檔案)
- ✅ 整合到 AI Prompt (大量文字 context)

---

## 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

主要套件:
- `feedparser` - RSS/Atom 解析
- `beautifulsoup4` - HTML 解析
- `requests` - HTTP 請求
- `lxml` - XML 處理

### 2. 基本使用

```python
from worldmonitor_integration import WorldMonitorAI

# 初始化
ai = WorldMonitorAI()

# 抓取最新新聞
news = ai.fetch_latest_news(
    hours=24,              # 最近 24 小時
    categories=['crypto'], # 只要加密貨幣新聞
    include_content=False  # 不爬完整內容 (更快)
)

print(f"共 {len(news)} 條新聞")

# 建構 AI Prompt
user_query = "比特幣最近的走勢如何?"
prompt = ai.ask(user_query)

# 送給你的 AI 模型 (DeepSeek/GPT/Claude)
response = your_ai_model.generate(prompt)
print(response)
```

---

## 詳細使用

### 抓取新聞

```python
from worldmonitor_integration import WorldMonitorAI

ai = WorldMonitorAI(cache_dir="cache")

# 方法 1: 快速抓取 (只有標題和摘要)
news = ai.fetch_latest_news(
    hours=24,
    categories=['crypto'],
    include_content=False  # 快速模式
)

# 方法 2: 完整爬取 (包含完整文章內容)
news = ai.fetch_latest_news(
    hours=12,
    categories=['crypto', 'finance'],
    include_content=True  # 會爬取每篇文章,比較慢
)

# 方法 3: 所有分類
news = ai.fetch_latest_news(
    hours=24,
    categories=None  # 所有分類: crypto, finance, taiwan
)
```

### 過濾新聞

```python
# 過濾 Bitcoin 相關
bitcoin_news = ai.filter_by_keywords(['bitcoin', 'btc'])

# 過濾 Ethereum 相關
ethereum_news = ai.filter_by_keywords(['ethereum', 'eth', 'vitalik'])

# 過濾監管相關
regulation_news = ai.filter_by_keywords(['sec', 'regulation', 'lawsuit'])

# 獲取高風險警報
alerts = ai.get_alerts()
for item in alerts:
    threat = item['threat']
    print(f"[{threat['level'].upper()}] {item['title']}")
```

### 建構 AI Prompt

```python
# 方法 1: 使用 ask() 方法 (最簡單)
user_query = "分析 BTC 短期走勢"
prompt = ai.ask(user_query, auto_fetch_news=True)

# 方法 2: 手動建構新聞_context = ai.build_news_context(
    news_list=news,
    max_items=20,
    include_full_content=True
)

prompt = ai.build_ai_prompt(
    user_query="分析 BTC",
    news_list=news,
    market_data={
        'price': 65000,
        'change_24h': +2.5
    }
)

# 送給 AI 模型
response = deepseek_api.generate(prompt)  # 你的 AI 呼叫
```

### 整合到你的系統

```python
# 在你的交易系統中
import streamlit as st
from worldmonitor_integration import WorldMonitorAI

# Streamlit 使用範例
if 'worldmonitor' not in st.session_state:
    st.session_state.worldmonitor = WorldMonitorAI()

ai = st.session_state.worldmonitor

# 在 Sidebar 加入新聞更新按鈕
with st.sidebar:
    if st.button("🔄 更新新聞"):
        with st.spinner("抓取中..."):
            news = ai.fetch_latest_news(hours=24, categories=['crypto'])
            st.success(f"共 {len(news)} 條新聞")

# 在 AI 分析中使用
if st.button("🤖 獲取 AI 分析"):
    user_query = st.text_input("問題")
    
    # 建構 Prompt
    prompt = ai.ask(user_query, auto_fetch_news=True)
    
    # 呼叫你的 AI
    response = your_ai_api.generate(prompt)
    
    st.write(response)
```

---

## 新聞來源配置

### 預設來源 (in `core/news_aggregator.py`)

**加密貨幣來源**:
- CoinDesk - https://www.coindesk.com/arc/outboundfeeds/rss/
- CryptoSlate - https://cryptoslate.com/feed/
- Cointelegraph - https://cointelegraph.com/rss
- Bitcoin Magazine - https://bitcoinmagazine.com/.rss/full/
- The Block - https://www.theblock.co/rss.xml
- Decrypt - https://decrypt.co/feed

**金融市場**:
- Reuters Markets
- Bloomberg Crypto
- Financial Times

**台灣財經**:
- 鉅亨網
- 經濟日報

### 自訂來源

在 `core/news_aggregator.py` 中修改 `NewsAggregator.FEEDS`:

```python
FEEDS = [
    {
        'name': '你的來源名稱',
        'url': 'https://example.com/rss',
        'category': 'crypto',  # crypto/finance/taiwan
        'lang': 'en',          # en/zh-TW
        'priority': 'high'     # high/medium/low
    },
    # ... 更多來源
]
```

---

## 威脅等級分類

系統會自動根據關鍵字分類新聞的風險等級:

### Critical (致命)
- hack, hacked, exploit, rugpull, scam, stolen
- crash, collapse, bankruptcy, insolvent, fraud
- 駭客, 詐騙, 崩盤, 破產

### High (高風險)
- warning, alert, risk, danger, threat, vulnerable
- security, breach, attack, suspend, halt
- 警告, 風險, 暫停, 攻擊

### Medium (中等)
- concern, issue, problem, delay, investigate
- question, doubt, uncertain
- 疑慮, 問題, 延遲, 調查

### Low (低風險)
- 其他所有新聞

---

## Prompt 結構

系統生成的 Prompt 結構如下:

```
=== SYSTEM ROLE ===
你是專業的加密貨幣交易分析師...

=== CURRENT MARKET DATA ===
- Price: $65,000
- 24h Change: +2.5%

=== WORLDMONITOR NEWS CONTEXT ===

## HIGH-PRIORITY ALERTS
[高風險新聞列表,包含完整內容]

## REGULAR NEWS
[一般新聞列表,包含完整內容]

## AI ANALYSIS GUIDELINES
- 如何使用這些 context
- 如何引用新聞
- 風險提示

=== USER QUERY ===
比特幣最近的走勢如何?

=== YOUR TASK ===
請提供分析...
```

**優點**:
1. 分離高風險新聞
2. 包含完整文章內容
3. 清晰的結構化 context
4. 提供 AI 使用指引
5. 支援大量文字 (20+ 篇新聞,每篇 2000 字)

---

## 效能最佳化

### 1. 使用快取

```python
ai = WorldMonitorAI()

# 第一次抓取
news = ai.fetch_latest_news(hours=24)

# 5分鐘內再次呼叫會使用快取
news = ai.fetch_latest_news(hours=24)  # 快速返回

# 強制重新抓取
news = ai.fetch_latest_news(hours=24, force_refresh=True)
```

### 2. 不爬完整內容 (更快)

```python
# 快速模式: 只有 RSS 摘要
news = ai.fetch_latest_news(
    hours=24,
    include_content=False  # 5-10秒
)

# 完整模式: 爬取每篇文章
news = ai.fetch_latest_news(
    hours=24,
    include_content=True  # 30-60秒
)
```

### 3. 限制新聞數量

```python
# 在 build_news_context 中限制
context = ai.build_news_context(
    news_list=news,
    max_items=10,           # 只用前 10 條
    max_content_length=1000 # 每條最多 1000 字
)
```

### 4. 儲存與載入

```python
# 儲存新聞
ai.fetch_latest_news(hours=24)
ai.save_news("latest_news.json")

# 下次直接載入
ai2 = WorldMonitorAI()
ai2.load_news("latest_news.json")
```

---

## 常見問題

### Q: 為什麼抓取失敗?

A: 可能原因:
1. RSS 來源暫時不可用
2. 網路連線問題
3. 超過失敗次數限制 (自動冷卻 5 分鐘)

解決方法:
```python
# 查看失敗統計
stats = ai.get_statistics()
print(ai.builder.format_statistics(stats))

# 重試
ai.fetch_latest_news(hours=24, force_refresh=True)
```

### Q: 如何不開啟瀏覽器視窗?

A: 系統使用 `requests` 不是 `playwright`,不會開啟視窗。

如果你的其他爬虫用到 `playwright`,設定:
```python
browser = playwright.chromium.launch(headless=True)
```

### Q: Prompt 太長怎麼辦?

A: 調整參數:
```python
context = ai.build_news_context(
    news_list=news[:10],        # 只用前 10 條
    max_items=10,
    max_content_length=800,     # 每條最多 800 字
    include_full_content=False  # 不包含完整內容
)
```

### Q: 如何自訂 AI 指令?

A:
```python
custom_instruction = """
你是專業的加密貨幣短線交易員。
專長: 15分鐘線技術分析。
風格: 直接、簡潔、可執行。
"""

prompt = ai.build_ai_prompt(
    user_query="給我 BTC 短線訊號",
    system_instruction=custom_instruction
)
```

---

## 進階功能

### 多語言支援

```python
# 繁中新聞
zh_news = ai.filter_by_keywords(['比特幣', '以太幣'])

# 英文新聞
en_news = ai.filter_by_keywords(['bitcoin', 'ethereum'])
```

### 威脇等級過濾

```python
# 只看高風險新聞
high_risk = [n for n in news if n['threat']['level'] in ['critical', 'high']]

# 只看低風險新聞
low_risk = [n for n in news if n['threat']['level'] in ['low', 'medium']]
```

### 時間範圍過濾

```python
from datetime import datetime, timedelta

# 最近 6 小時
cutoff = datetime.now() - timedelta(hours=6)
recent = [n for n in news if n['published'] >= cutoff]
```

---

## 測試

```bash
# 執行測試
python worldmonitor_integration.py

# 測試單個模組
python core/news_aggregator.py
python core/ai_prompt_builder.py
```

---

## 責任聲明

1. **資訊來源**: 所有新聞均來自公開 RSS feeds
2. **准確性**: 系統不保證新聞內容的準確性
3. **投資風險**: 不構成投資建議,僅供參考
4. **爬虫合法性**: 請遵守各網站的 robots.txt 與服務條款

---

## 資源

- **WorldMonitor 原始專案**: https://github.com/koala73/worldmonitor
- **RSS Feed 規格**: https://www.rssboard.org/rss-specification
- **Beautiful Soup 文檔**: https://www.crummy.com/software/BeautifulSoup/

---

## 貳獻

- WorldMonitor team for the original news aggregation logic
- 所有 RSS feed 提供者

---

**更新時間**: 2026-03-09  
**版本**: 1.0.0  
**作者**: caizongxun
