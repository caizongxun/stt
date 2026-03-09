"""新聞聚合系統 - 整合 WorldMonitor 邏輯

功能:
1. RSS feed 抓取 (支援 RSS/Atom)
2. 完整文章內容爬蟲 (使用 requests,不開啟瀏覽器)
3. 圖片提取 (多種格式支援)
4. 時間過濾與排序
5. 關鍵字搜尋
6. 快取機制 (記憶體+文件)
7. WorldMonitor 威脅等級分類
"""

import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import json
from pathlib import Path
import re


class NewsAggregator:
    """新聞聚合器"""
    
    # RSS Feed 來源配置 (參考 WorldMonitor)
    FEEDS = [
        # === 加密貨幣核心來源 ===
        {
            'name': 'CoinDesk',
            'url': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'category': 'crypto',
            'lang': 'en',
            'priority': 'high'
        },
        {
            'name': 'CryptoSlate',
            'url': 'https://cryptoslate.com/feed/',
            'category': 'crypto',
            'lang': 'en',
            'priority': 'high'
        },
        {
            'name': 'Cointelegraph',
            'url': 'https://cointelegraph.com/rss',
            'category': 'crypto',
            'lang': 'en',
            'priority': 'high'
        },
        {
            'name': 'Bitcoin Magazine',
            'url': 'https://bitcoinmagazine.com/.rss/full/',
            'category': 'crypto',
            'lang': 'en',
            'priority': 'medium'
        },
        {
            'name': 'The Block',
            'url': 'https://www.theblock.co/rss.xml',
            'category': 'crypto',
            'lang': 'en',
            'priority': 'high'
        },
        {
            'name': 'Decrypt',
            'url': 'https://decrypt.co/feed',
            'category': 'crypto',
            'lang': 'en',
            'priority': 'medium'
        },
        
        # === 金融市場 ===
        {
            'name': 'Reuters Markets',
            'url': 'https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best',
            'category': 'finance',
            'lang': 'en',
            'priority': 'high'
        },
        {
            'name': 'Bloomberg Crypto',
            'url': 'https://www.bloomberg.com/crypto/rss',
            'category': 'finance',
            'lang': 'en',
            'priority': 'high'
        },
        {
            'name': 'Financial Times',
            'url': 'https://www.ft.com/?format=rss',
            'category': 'finance',
            'lang': 'en',
            'priority': 'medium'
        },
        
        # === 台灣財經 ===
        {
            'name': '鉅亨網',
            'url': 'https://news.cnyes.com/rss/tw_stock.xml',
            'category': 'taiwan',
            'lang': 'zh-TW',
            'priority': 'medium'
        },
        {
            'name': '經濟日報',
            'url': 'https://money.udn.com/rssfeed/news/1001/5591/5599?ch=fb_share',
            'category': 'taiwan',
            'lang': 'zh-TW',
            'priority': 'medium'
        },
    ]
    
    # 威脅等級關鍵字 (參考 WorldMonitor)
    THREAT_KEYWORDS = {
        'critical': [
            'hack', 'hacked', 'exploit', 'rugpull', 'scam', 'stolen',
            'crash', 'collapse', 'bankruptcy', 'insolvent', 'fraud',
            '駭客', '詐騙', '崩盤', '破產'
        ],
        'high': [
            'warning', 'alert', 'risk', 'danger', 'threat', 'vulnerable',
            'security', 'breach', 'attack', 'suspend', 'halt',
            '警告', '風險', '暫停', '攻擊'
        ],
        'medium': [
            'concern', 'issue', 'problem', 'delay', 'investigate',
            'question', 'doubt', 'uncertain',
            '疑慮', '問題', '延遲', '調查'
        ]
    }
    
    def __init__(self, cache_dir: str = "cache"):
        """初始化
        
        Args:
            cache_dir: 快取目錄
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # 記憶體快取
        self.memory_cache = {}
        self.cache_duration = 1800  # 30分鐘
        
        # 失敗計數 (參考 WorldMonitor 的 cooldown 機制)
        self.failure_count = {}
        self.max_failures = 3
    
    def classify_threat(self, title: str, content: str = "") -> Dict:
        """分類威脅等級 (參考 WorldMonitor)
        
        Args:
            title: 標題
            content: 內容
            
        Returns:
            威脅資訊字典
        """
        text = f"{title} {content}".lower()
        
        # 檢查關鍵字
        for level in ['critical', 'high', 'medium']:
            for keyword in self.THREAT_KEYWORDS[level]:
                if keyword.lower() in text:
                    return {
                        'level': level,
                        'keyword': keyword,
                        'confidence': 0.8
                    }
        
        return {
            'level': 'low',
            'keyword': None,
            'confidence': 0.5
        }
    
    def fetch_feed(self, feed_config: Dict) -> List[Dict]:
        """抓取單個 RSS feed
        
        Args:
            feed_config: feed 配置字典
            
        Returns:
            新聞列表
        """
        feed_name = feed_config['name']
        
        # 檢查失敗次數
        if self.failure_count.get(feed_name, 0) >= self.max_failures:
            print(f"⏸️ 跳過 {feed_name} (失敗次數過多)")
            return []
        
        try:
            # 檢查快取
            cache_key = feed_config['url']
            if cache_key in self.memory_cache:
                cached_time, cached_data = self.memory_cache[cache_key]
                if time.time() - cached_time < self.cache_duration:
                    print(f"✅ 使用快取: {feed_name}")
                    return cached_data
            
            print(f"🔄 抓取: {feed_name}...")
            
            # 解析 RSS
            feed = feedparser.parse(feed_config['url'])
            
            if not feed.entries:
                raise Exception("無法解析 feed")
            
            items = []
            
            for entry in feed.entries[:10]:  # 每個來源取前10條
                title = entry.get('title', '').strip()
                link = entry.get('link', '').strip()
                
                if not title or not link:
                    continue
                
                summary = self._clean_html(entry.get('summary', ''))
                threat = self.classify_threat(title, summary)
                
                item = {
                    'source': feed_name,
                    'category': feed_config['category'],
                    'lang': feed_config['lang'],
                    'priority': feed_config.get('priority', 'medium'),
                    'title': title,
                    'link': link,
                    'summary': summary,
                    'published': self._parse_date(entry),
                    'image_url': self._extract_image(entry),
                    'threat': threat,
                    'is_alert': threat['level'] in ['critical', 'high']
                }
                
                items.append(item)
            
            # 存入快取
            self.memory_cache[cache_key] = (time.time(), items)
            
            # 重置失敗計數
            if feed_name in self.failure_count:
                del self.failure_count[feed_name]
            
            print(f"✅ 成功: {feed_name} ({len(items)} 條)")
            return items
            
        except Exception as e:
            print(f"❌ 抓取失敗 {feed_name}: {e}")
            self.failure_count[feed_name] = self.failure_count.get(feed_name, 0) + 1
            return []
    
    def _parse_date(self, entry) -> datetime:
        """解析日期"""
        for date_field in ['published_parsed', 'updated_parsed']:
            if hasattr(entry, date_field):
                time_struct = getattr(entry, date_field)
                if time_struct:
                    return datetime(*time_struct[:6])
        return datetime.now()
    
    def _extract_image(self, entry) -> Optional[str]:
        """提取圖片 URL (參考 WorldMonitor 的 extractImageUrl)"""
        # 1. media:content (Yahoo MRSS)
        if hasattr(entry, 'media_content') and entry.media_content:
            for media in entry.media_content:
                url = media.get('url')
                if url and self._is_valid_image_url(url):
                    return url
        
        # 2. media:thumbnail
        if hasattr(entry, 'media_thumbnail') and entry.media_thumbnail:
            for thumb in entry.media_thumbnail:
                url = thumb.get('url')
                if url:
                    return url
        
        # 3. enclosure
        if hasattr(entry, 'enclosures') and entry.enclosures:
            for enc in entry.enclosures:
                if enc.get('type', '').startswith('image'):
                    url = enc.get('href')
                    if url:
                        return url
        
        # 4. 從 summary/content 中提取 <img>
        for field in ['summary', 'content']:
            if hasattr(entry, field):
                text = getattr(entry, field)
                if isinstance(text, list):
                    text = text[0].get('value', '')
                
                soup = BeautifulSoup(str(text), 'html.parser')
                img = soup.find('img')
                if img and img.get('src'):
                    return img['src']
        
        return None
    
    def _is_valid_image_url(self, url: str) -> bool:
        """檢查是否為有效圖片 URL"""
        image_extensions = r'\.(jpg|jpeg|png|gif|webp|avif|svg)(\?|$)'
        return bool(re.search(image_extensions, url, re.IGNORECASE))
    
    def _clean_html(self, text: str) -> str:
        """清理 HTML 標籤"""
        if not text:
            return ""
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text().strip()
    
    def scrape_article_content(self, url: str, timeout: int = 10) -> Optional[str]:
        """爬取文章完整內容 (使用 requests,不開啟瀏覽器)
        
        Args:
            url: 文章 URL
            timeout: 超時時間(秒)
            
        Returns:
            文章內容
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 移除不需要的元素
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 
                           'aside', 'iframe', 'form', 'button', 'noscript']):
                tag.decompose()
            
            # 嘗試多種常見文章選擇器
            content_selectors = [
                'article',
                '.article-content',
                '.article-body',
                '.post-content',
                '.entry-content',
                '[itemprop="articleBody"]',
                '.story-body',
                '.content-body',
                'main article',
                'main'
            ]
            
            article_text = None
            for selector in content_selectors:
                article = soup.select_one(selector)
                if article:
                    # 提取所有段落
                    paragraphs = article.find_all('p')
                    article_text = '\n\n'.join([
                        p.get_text().strip() 
                        for p in paragraphs 
                        if p.get_text().strip()
                    ])
                    
                    if len(article_text) > 200:  # 確保有足夠內容
                        break
            
            # 如果找不到,就抓所有 p 標籤
            if not article_text or len(article_text) < 200:
                paragraphs = soup.find_all('p')
                article_text = '\n\n'.join([
                    p.get_text().strip() 
                    for p in paragraphs 
                    if p.get_text().strip() and len(p.get_text().strip()) > 20
                ])
            
            return article_text if article_text else None
            
        except requests.Timeout:
            print(f"⏱️ 超時: {url}")
            return None
        except Exception as e:
            print(f"❌ 爬取失敗 {url}: {e}")
            return None
    
    def fetch_all_news(
        self, 
        hours: int = 24, 
        include_content: bool = False,
        categories: Optional[List[str]] = None,
        max_per_feed: int = 10
    ) -> List[Dict]:
        """抓取所有新聞
        
        Args:
            hours: 抓取最近幾小時的新聞
            include_content: 是否爬取完整文章內容 (會比較慢)
            categories: 過濾分類 (None=全部)
            max_per_feed: 每個 feed 最多取幾條
            
        Returns:
            新聞列表
        """
        all_news = []
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        print(f"\n{'='*60}")
        print(f"🔍 開始抓取新聞 (最近 {hours} 小時)")
        print(f"{'='*60}\n")
        
        # 過濾 feeds
        feeds_to_fetch = self.FEEDS
        if categories:
            feeds_to_fetch = [
                f for f in self.FEEDS 
                if f['category'] in categories
            ]
        
        for feed_config in feeds_to_fetch:
            items = self.fetch_feed(feed_config)
            
            for item in items[:max_per_feed]:
                # 過濾時間
                if item['published'] < cutoff_time:
                    continue
                
                # 是否需要爬取完整內容
                if include_content:
                    print(f"  🕷️ 爬取文章: {item['title'][:50]}...")
                    content = self.scrape_article_content(item['link'])
                    
                    if content:
                        item['full_content'] = content
                        # 重新分類威脅等級 (使用完整內容)
                        threat = self.classify_threat(item['title'], content)
                        item['threat'] = threat
                        item['is_alert'] = threat['level'] in ['critical', 'high']
                    else:
                        item['full_content'] = item['summary']
                    
                    time.sleep(0.5)  # 禮貌性延遲
                else:
                    item['full_content'] = item['summary']
                
                all_news.append(item)
        
        # 按時間排序 (最新在前)
        all_news.sort(key=lambda x: x['published'], reverse=True)
        
        print(f"\n{'='*60}")
        print(f"✅ 共抓取 {len(all_news)} 條新聞")
        
        # 統計警報數量
        alert_count = sum(1 for item in all_news if item['is_alert'])
        if alert_count > 0:
            print(f"⚠️ 其中 {alert_count} 條為高風險新聞")
        
        print(f"{'='*60}\n")
        
        return all_news
    
    def filter_by_keywords(
        self, 
        news_list: List[Dict], 
        keywords: List[str],
        search_in: List[str] = ['title', 'summary']
    ) -> List[Dict]:
        """根據關鍵字過濾新聞
        
        Args:
            news_list: 新聞列表
            keywords: 關鍵字列表
            search_in: 搜尋範圍 ['title', 'summary', 'full_content']
            
        Returns:
            過濾後的新聞列表
        """
        filtered = []
        
        for item in news_list:
            # 建構搜尋文本
            search_text = ""
            if 'title' in search_in:
                search_text += item.get('title', '') + " "
            if 'summary' in search_in:
                search_text += item.get('summary', '') + " "
            if 'full_content' in search_in:
                search_text += item.get('full_content', '') + " "
            
            search_text = search_text.lower()
            
            # 檢查關鍵字
            if any(keyword.lower() in search_text for keyword in keywords):
                filtered.append(item)
        
        return filtered
    
    def get_alerts(self, news_list: List[Dict]) -> List[Dict]:
        """獲取高風險新聞
        
        Args:
            news_list: 新聞列表
            
        Returns:
            警報新聞列表
        """
        return [item for item in news_list if item['is_alert']]
    
    def save_to_json(self, news_list: List[Dict], filename: str = "news.json"):
        """存檔為 JSON
        
        Args:
            news_list: 新聞列表
            filename: 檔案名稱
        """
        filepath = self.cache_dir / filename
        
        # 轉換 datetime 為字串
        data_to_save = []
        for item in news_list:
            item_copy = item.copy()
            item_copy['published'] = item['published'].isoformat()
            data_to_save.append(item_copy)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        
        print(f"💾 已存檔: {filepath}")
    
    def load_from_json(self, filename: str = "news.json") -> List[Dict]:
        """從 JSON 載入
        
        Args:
            filename: 檔案名稱
            
        Returns:
            新聞列表
        """
        filepath = self.cache_dir / filename
        
        if not filepath.exists():
            return []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 轉換字串回 datetime
        for item in data:
            item['published'] = datetime.fromisoformat(item['published'])
        
        return data


if __name__ == "__main__":
    # 測試
    aggregator = NewsAggregator()
    
    # 抓取最近24小時的加密貨幣新聞
    news = aggregator.fetch_all_news(
        hours=24,
        include_content=False,
        categories=['crypto']
    )
    
    # 過濾比特幣相關
    bitcoin_news = aggregator.filter_by_keywords(
        news,
        keywords=['bitcoin', 'btc', 'ethereum', 'eth']
    )
    
    print(f"\n比特幣相關新聞: {len(bitcoin_news)} 條")
    
    # 獲取警報
    alerts = aggregator.get_alerts(bitcoin_news)
    if alerts:
        print(f"⚠️ 高風險新聞: {len(alerts)} 條")
        for item in alerts[:3]:
            print(f"  - [{item['threat']['level'].upper()}] {item['title']}")
    
    # 存檔
    aggregator.save_to_json(bitcoin_news, "bitcoin_news.json")
