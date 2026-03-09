"""新聞聚合系統 - 無瀏覽器視窗版本

功能:
1. RSS feed 抓取
2. 完整文章內容爬蟲 (使用 requests,不開啟瀏覽器)
3. 圖片提取
4. 時間過濾
5. 關鍵字搜尋
6. 快取機制
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
    
    # RSS Feed 來源配置
    FEEDS = [
        # 加密貨幣專用
        {
            'name': 'CoinDesk',
            'url': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'category': 'crypto',
            'lang': 'en'
        },
        {
            'name': 'CryptoSlate',
            'url': 'https://cryptoslate.com/feed/',
            'category': 'crypto',
            'lang': 'en'
        },
        {
            'name': 'Cointelegraph',
            'url': 'https://cointelegraph.com/rss',
            'category': 'crypto',
            'lang': 'en'
        },
        {
            'name': 'Bitcoin Magazine',
            'url': 'https://bitcoinmagazine.com/.rss/full/',
            'category': 'crypto',
            'lang': 'en'
        },
        {
            'name': 'The Block',
            'url': 'https://www.theblock.co/rss.xml',
            'category': 'crypto',
            'lang': 'en'
        },
        # 金融市場
        {
            'name': 'Reuters Markets',
            'url': 'https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best',
            'category': 'finance',
            'lang': 'en'
        },
        # 台灣財經
        {
            'name': '鉅亨網',
            'url': 'https://news.cnyes.com/rss/tw_stock.xml',
            'category': 'taiwan',
            'lang': 'zh-TW'
        },
    ]
    
    def __init__(self, cache_dir: str = "cache"):
        """初始化
        
        Args:
            cache_dir: 快取目錄
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        # 記憶體快取
        self.memory_cache = {}
        self.cache_duration = 1800  # 30分鐘
    
    def fetch_feed(self, feed_config: Dict) -> List[Dict]:
        """抓取單個 RSS feed
        
        Args:
            feed_config: feed 配置字典
            
        Returns:
            新聞列表
        """
        try:
            # 檢查快取
            cache_key = feed_config['url']
            if cache_key in self.memory_cache:
                cached_time, cached_data = self.memory_cache[cache_key]
                if time.time() - cached_time < self.cache_duration:
                    print(f"✅ 使用快取: {feed_config['name']}")
                    return cached_data
            
            print(f"🔄 抓取: {feed_config['name']}...")
            
            # 解析 RSS
            feed = feedparser.parse(feed_config['url'])
            items = []
            
            for entry in feed.entries[:10]:  # 每個來源取前10條
                item = {
                    'source': feed_config['name'],
                    'category': feed_config['category'],
                    'lang': feed_config['lang'],
                    'title': entry.get('title', '').strip(),
                    'link': entry.get('link', '').strip(),
                    'summary': self._clean_html(entry.get('summary', '')),
                    'published': self._parse_date(entry),
                    'image_url': self._extract_image(entry)
                }
                
                if item['title'] and item['link']:
                    items.append(item)
            
            # 存入快取
            self.memory_cache[cache_key] = (time.time(), items)
            
            print(f"✅ 成功: {feed_config['name']} ({len(items)} 條)")
            return items
            
        except Exception as e:
            print(f"❌ 抓取失敗 {feed_config['name']}: {e}")
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
        """提取圖片 URL"""
        # 1. media:content (Yahoo MRSS)
        if hasattr(entry, 'media_content') and entry.media_content:
            return entry.media_content[0].get('url')
        
        # 2. media:thumbnail
        if hasattr(entry, 'media_thumbnail') and entry.media_thumbnail:
            return entry.media_thumbnail[0].get('url')
        
        # 3. enclosure
        if hasattr(entry, 'enclosures') and entry.enclosures:
            for enc in entry.enclosures:
                if enc.get('type', '').startswith('image'):
                    return enc.get('href')
        
        # 4. 從 summary 中提取 <img>
        if hasattr(entry, 'summary'):
            soup = BeautifulSoup(entry.summary, 'html.parser')
            img = soup.find('img')
            if img and img.get('src'):
                return img['src']
        
        return None
    
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
                           'aside', 'iframe', 'form', 'button']):
                tag.decompose()
            
            # 嘗試多種常見文章選擇器
            content_selectors = [
                'article',
                '.article-content',
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
                    if p.get_text().strip()
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
        categories: Optional[List[str]] = None
    ) -> List[Dict]:
        """抓取所有新聞
        
        Args:
            hours: 抓取最近幾小時的新聞
            include_content: 是否爬取完整文章內容 (會比較慢)
            categories: 過濾分類 (None=全部)
            
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
            
            for item in items:
                # 過濾時間
                if item['published'] < cutoff_time:
                    continue
                
                # 是否需要爬取完整內容
                if include_content:
                    print(f"  🕷️ 爬取文章: {item['title'][:50]}...")
                    content = self.scrape_article_content(item['link'])
                    item['full_content'] = content if content else item['summary']
                    time.sleep(1)  # 禮貌性延遲
                else:
                    item['full_content'] = item['summary']
                
                all_news.append(item)
        
        # 按時間排序
        all_news.sort(key=lambda x: x['published'], reverse=True)
        
        print(f"\n{'='*60}")
        print(f"✅ 共抓取 {len(all_news)} 條新聞")
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
        include_content=False,  # 測試時先不爬完整內容
        categories=['crypto']
    )
    
    # 過濾比特幣相關
    bitcoin_news = aggregator.filter_by_keywords(
        news,
        keywords=['bitcoin', 'btc', 'ethereum', 'eth']
    )
    
    print(f"\n比特幣相關新聞: {len(bitcoin_news)} 條")
    
    for idx, item in enumerate(bitcoin_news[:5], 1):
        print(f"\n{idx}. {item['title']}")
        print(f"   來源: {item['source']}")
        print(f"   時間: {item['published'].strftime('%Y-%m-%d %H:%M')}")
