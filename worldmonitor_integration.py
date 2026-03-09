#!/usr/bin/env python3
"""WorldMonitor Integration - Easy-to-use wrapper

這個檔案提供簡單的介面來整合 WorldMonitor 新聞系統到你的 AI 交易系統中

使用方式:
    from worldmonitor_integration import WorldMonitorAI
    
    ai = WorldMonitorAI()
    response = ai.ask("比特幣最近走勢如何?")
    print(response)
"""

from core.news_aggregator import NewsAggregator
from core.ai_prompt_builder import AIPromptBuilder
from typing import List, Dict, Optional
from datetime import datetime
import json


class WorldMonitorAI:
    """整合 WorldMonitor 新聞的 AI 助手"""
    
    def __init__(self, cache_dir: str = "cache"):
        """初始化
        
        Args:
            cache_dir: 快取目錄
        """
        self.aggregator = NewsAggregator(cache_dir=cache_dir)
        self.builder = AIPromptBuilder()
        self.latest_news = []
        self.last_update = None
    
    def fetch_latest_news(
        self,
        hours: int = 24,
        include_content: bool = False,
        categories: Optional[List[str]] = None,
        force_refresh: bool = False
    ) -> List[Dict]:
        """抓取最新新聞
        
        Args:
            hours: 抓取最近幾小時的新聞
            include_content: 是否爬取完整文章內容 (會比較慢)
            categories: 過濾分類 ['crypto', 'finance', 'taiwan']
            force_refresh: 強制重新抓取
            
        Returns:
            新聞列表
        """
        # 快取檢查 (5分鐘內不重複抓取)
        if not force_refresh and self.latest_news and self.last_update:
            elapsed = (datetime.now() - self.last_update).seconds
            if elapsed < 300:  # 5分鐘
                print(f"✅ 使用快取的新聞 (已經過 {elapsed}秒)")
                return self.latest_news
        
        print(f"\n{'='*60}")
        print("🌍 WorldMonitor 新聞更新")
        print(f"{'='*60}\n")
        
        self.latest_news = self.aggregator.fetch_all_news(
            hours=hours,
            include_content=include_content,
            categories=categories
        )
        
        self.last_update = datetime.now()
        
        return self.latest_news
    
    def build_news_context(
        self,
        news_list: Optional[List[Dict]] = None,
        max_items: int = 20,
        include_full_content: bool = True
    ) -> str:
        """建構新聞 context
        
        Args:
            news_list: 新聞列表 (None = 使用 latest_news)
            max_items: 最多包含幾條新聞
            include_full_content: 是否包含完整內容
            
        Returns:
            格式化的新聞 context
        """
        if news_list is None:
            news_list = self.latest_news
        
        if not news_list:
            print("⚠️ 沒有可用的新聞,請先呼叫 fetch_latest_news()")
            news_list = []
        
        return self.builder.build_news_context(
            news_list=news_list,
            max_items=max_items,
            include_full_content=include_full_content,
            highlight_alerts=True
        )
    
    def build_ai_prompt(
        self,
        user_query: str,
        news_list: Optional[List[Dict]] = None,
        market_data: Optional[Dict] = None,
        system_instruction: Optional[str] = None
    ) -> str:
        """建構完整的 AI prompt
        
        Args:
            user_query: 用戶問題
            news_list: 新聞列表 (None = 使用 latest_news)
            market_data: 市場數據
            system_instruction: 自訂系統指令
            
        Returns:
            完整 prompt
        """
        if news_list is None:
            news_list = self.latest_news
        
        news_context = self.build_news_context(news_list)
        
        return self.builder.build_complete_prompt(
            user_query=user_query,
            news_context=news_context,
            market_data=market_data,
            system_instruction=system_instruction
        )
    
    def ask(
        self,
        user_query: str,
        auto_fetch_news: bool = True,
        market_data: Optional[Dict] = None,
        **kwargs
    ) -> str:
        """直接問 AI (不實際呼叫 AI,只返回 prompt)
        
        這個方法返回完整的 prompt,你需要自己把它送給 DeepSeek/GPT 等模型
        
        Args:
            user_query: 用戶問題
            auto_fetch_news: 自動抓取最新新聞
            market_data: 市場數據
            **kwargs: 傳遞給 fetch_latest_news 的參數
            
        Returns:
            完整的 AI prompt (送給模型使用)
        """
        # 自動抓取新聞
        if auto_fetch_news and not self.latest_news:
            self.fetch_latest_news(**kwargs)
        
        # 建構 prompt
        prompt = self.build_ai_prompt(
            user_query=user_query,
            market_data=market_data
        )
        
        return prompt
    
    def get_alerts(self, news_list: Optional[List[Dict]] = None) -> List[Dict]:
        """獲取高風險新聞
        
        Args:
            news_list: 新聞列表 (None = 使用 latest_news)
            
        Returns:
            警報新聞列表
        """
        if news_list is None:
            news_list = self.latest_news
        
        return self.aggregator.get_alerts(news_list)
    
    def filter_by_keywords(
        self,
        keywords: List[str],
        news_list: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """根據關鍵字過濾新聞
        
        Args:
            keywords: 關鍵字列表
            news_list: 新聞列表 (None = 使用 latest_news)
            
        Returns:
            過濾後的新聞列表
        """
        if news_list is None:
            news_list = self.latest_news
        
        return self.aggregator.filter_by_keywords(news_list, keywords)
    
    def get_statistics(self, news_list: Optional[List[Dict]] = None) -> Dict:
        """獲取統計資訊
        
        Args:
            news_list: 新聞列表 (None = 使用 latest_news)
            
        Returns:
            統計字典
        """
        if news_list is None:
            news_list = self.latest_news
        
        return self.builder.get_statistics(news_list)
    
    def save_news(self, filename: str = "latest_news.json"):
        """儲存新聞到檔案
        
        Args:
            filename: 檔案名稱
        """
        if not self.latest_news:
            print("⚠️ 沒有新聞可儲存")
            return
        
        self.aggregator.save_to_json(self.latest_news, filename)
    
    def load_news(self, filename: str = "latest_news.json") -> List[Dict]:
        """從檔案載入新聞
        
        Args:
            filename: 檔案名稱
            
        Returns:
            新聞列表
        """
        self.latest_news = self.aggregator.load_from_json(filename)
        if self.latest_news:
            self.last_update = datetime.now()
        return self.latest_news


# ========== 使用範例 ==========

def example_basic_usage():
    """基本使用範例"""
    print("\n" + "="*80)
    print("🔵 範例 1: 基本使用")
    print("="*80 + "\n")
    
    # 初始化
    ai = WorldMonitorAI()
    
    # 抓取最新新聞
    news = ai.fetch_latest_news(
        hours=24,
        categories=['crypto'],  # 只要加密貨幣新聞
        include_content=False   # 不爬取完整內容 (更快)
    )
    
    print(f"\n✅ 共抓取 {len(news)} 條新聞")
    
    # 建構 AI prompt
    user_query = "比特幣最近的走勢如何?有什麼重大新聞影響?"
    prompt = ai.ask(user_query, auto_fetch_news=False)
    
    print(f"\n📝 Prompt 長度: {len(prompt):,} 字元")
    print(f"🤖 Token 估計: ~{len(prompt) // 4:,} tokens")
    
    # 儲存 prompt 供檢視
    with open("example_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)
    print("\n💾 Prompt 已儲存到 example_prompt.txt")
    
    # 顯示統計
    stats = ai.get_statistics()
    print("\n" + ai.builder.format_statistics(stats))


def example_filtered_news():
    """過濾新聞範例"""
    print("\n" + "="*80)
    print("🔵 範例 2: 過濾特定關鍵字新聞")
    print("="*80 + "\n")
    
    ai = WorldMonitorAI()
    ai.fetch_latest_news(hours=24, categories=['crypto'])
    
    # 過濾 Bitcoin 相關新聞
    bitcoin_news = ai.filter_by_keywords(['bitcoin', 'btc'])
    print(f"\n₿️ Bitcoin 相關新聞: {len(bitcoin_news)} 條")
    
    # 過濾 Ethereum 相關新聞
    ethereum_news = ai.filter_by_keywords(['ethereum', 'eth'])
    print(f"⛓️ Ethereum 相關新聞: {len(ethereum_news)} 條")
    
    # 獲取高風險新聞
    alerts = ai.get_alerts()
    if alerts:
        print(f"\n⚠️ 高風險警報: {len(alerts)} 條")
        for item in alerts[:3]:
            threat = item.get('threat', {})
            level = threat.get('level', 'unknown').upper()
            print(f"  [{level}] {item['title'][:60]}...")


def example_with_market_data():
    """整合市場數據範例"""
    print("\n" + "="*80)
    print("🔵 範例 3: 整合市場數據")
    print("="*80 + "\n")
    
    ai = WorldMonitorAI()
    ai.fetch_latest_news(hours=12, categories=['crypto'], include_content=False)
    
    # 模擬市場數據
    market_data = {
        'price': 65432.10,
        'change_24h': +2.34,
        'volume_24h': 28500000000,
        'timestamp': datetime.now().isoformat()
    }
    
    # 建構帶有市場數據的 prompt
    user_query = "根據當前價格和最新新聞,分析 BTC 短期走勢"
    prompt = ai.ask(user_query, auto_fetch_news=False, market_data=market_data)
    
    print(f"✅ Prompt 已建構 (包含市場數據)")
    print(f"📊 當前 BTC 價格: ${market_data['price']:,.2f}")
    print(f"📈 24h 變動: {market_data['change_24h']:+.2f}%")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🌍 WorldMonitor Integration - 使用範例")
    print("="*80)
    
    try:
        # 執行範例
        example_basic_usage()
        # example_filtered_news()
        # example_with_market_data()
        
        print("\n" + "="*80)
        print("✅ 所有範例執行完成!")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用戶中斷")
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
