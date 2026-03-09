"""AI Prompt Builder

建構給 AI 的完整 context,整合新聞資訊
"""

from typing import List, Dict, Optional
from datetime import datetime
import textwrap


class AIPromptBuilder:
    """AI Prompt 建構器"""
    
    @staticmethod
    def build_news_context(
        news_list: List[Dict],
        max_items: int = 15,
        max_content_length: int = 1500,
        include_full_content: bool = True
    ) -> str:
        """建構新聞 context
        
        Args:
            news_list: 新聞列表
            max_items: 最多包含幾條新聞
            max_content_length: 每條新聞內容最大長度
            include_full_content: 是否包含完整內容
            
        Returns:
            格式化的新聞 context
        """
        if not news_list:
            return "# 📰 新聞資訊\n\n目前沒有最新新聞。\n\n"
        
        context = "# 📰 最新加密貨幣與市場新聞動態\n\n"
        context += f"**更新時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        context += f"**新聞數量**: {len(news_list[:max_items])} 條\n\n"
        context += "=" * 80 + "\n\n"
        
        for idx, item in enumerate(news_list[:max_items], 1):
            context += f"## 【新聞 {idx}】{item['title']}\n\n"
            
            # 元資料
            context += f"- **來源**: {item['source']}\n"
            context += f"- **分類**: {item['category']}\n"
            context += f"- **時間**: {item['published'].strftime('%Y-%m-%d %H:%M')}\n"
            context += f"- **連結**: {item['link']}\n"
            
            if item.get('image_url'):
                context += f"- **圖片**: {item['image_url']}\n"
            
            context += "\n"
            
            # 內容
            if include_full_content and item.get('full_content'):
                content = item['full_content'][:max_content_length]
                if len(item['full_content']) > max_content_length:
                    content += "..."
                context += f"**完整內容**:\n{content}\n\n"
            else:
                summary = item['summary'][:500]
                if len(item['summary']) > 500:
                    summary += "..."
                context += f"**摘要**: {summary}\n\n"
            
            context += "-" * 80 + "\n\n"
        
        # 分析說明
        context += "\n# 📊 分析說明\n\n"
        context += textwrap.dedent("""
        以上是最新的全球加密貨幣與金融市場新聞。
        在回答用戶問題時,請考慮這些最新資訊對市場的影響。
        
        **重點關注**:
        1. 監管政策變化
        2. 主要交易所動態
        3. 技術發展與創新
        4. 市場情緒指標
        5. 重大事件與公告
        """).strip()
        
        context += "\n\n" + "=" * 80 + "\n\n"
        
        return context
    
    @staticmethod
    def build_complete_prompt(
        user_query: str,
        news_context: str,
        system_instruction: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> str:
        """建構完整的 AI prompt
        
        Args:
            user_query: 用戶問題
            news_context: 新聞 context
            system_instruction: 系統指令
            additional_context: 額外 context
            
        Returns:
            完整 prompt
        """
        prompt = ""
        
        # 系統指令
        if system_instruction:
            prompt += f"{system_instruction}\n\n"
            prompt += "=" * 80 + "\n\n"
        else:
            prompt += textwrap.dedent("""
            你是一位專業的加密貨幣交易分析師。
            
            **你的專長**:
            - 技術分析與圖表解讀
            - 基本面分析
            - 市場情緒分析
            - 風險管理
            - 交易策略制定
            
            **回答原則**:
            1. 基於事實和數據
            2. 考慮市場最新動態
            3. 提供多角度分析
            4. 明確標示風險
            5. 使用繁體中文回答
            """).strip()
            prompt += "\n\n" + "=" * 80 + "\n\n"
        
        # 新聞 context
        prompt += news_context
        
        # 額外 context
        if additional_context:
            prompt += additional_context + "\n\n"
            prompt += "=" * 80 + "\n\n"
        
        # 用戶問題
        prompt += "# 💬 用戶問題\n\n"
        prompt += f"{user_query}\n\n"
        prompt += "=" * 80 + "\n\n"
        
        # 回答提示
        prompt += "# 📝 請根據以上資訊回答\n\n"
        prompt += "請提供詳細的分析和建議。\n"
        
        return prompt
    
    @staticmethod
    def get_statistics(news_list: List[Dict]) -> Dict:
        """獲取新聞統計資訊
        
        Args:
            news_list: 新聞列表
            
        Returns:
            統計字典
        """
        if not news_list:
            return {
                'total': 0,
                'by_source': {},
                'by_category': {},
                'time_range': None
            }
        
        stats = {
            'total': len(news_list),
            'by_source': {},
            'by_category': {},
            'time_range': {
                'earliest': min(item['published'] for item in news_list),
                'latest': max(item['published'] for item in news_list)
            }
        }
        
        for item in news_list:
            # 按來源統計
            source = item['source']
            stats['by_source'][source] = stats['by_source'].get(source, 0) + 1
            
            # 按分類統計
            category = item['category']
            stats['by_category'][category] = stats['by_category'].get(category, 0) + 1
        
        return stats
    
    @staticmethod
    def format_statistics(stats: Dict) -> str:
        """格式化統計資訊
        
        Args:
            stats: 統計字典
            
        Returns:
            格式化字串
        """
        if stats['total'] == 0:
            return "沒有新聞資料"
        
        text = f"# 📊 新聞統計\n\n"
        text += f"**總數**: {stats['total']} 條\n\n"
        
        # 時間範圍
        if stats['time_range']:
            earliest = stats['time_range']['earliest'].strftime('%Y-%m-%d %H:%M')
            latest = stats['time_range']['latest'].strftime('%Y-%m-%d %H:%M')
            text += f"**時間範圍**: {earliest} ~ {latest}\n\n"
        
        # 按來源
        text += "**按來源**:\n"
        for source, count in sorted(stats['by_source'].items(), key=lambda x: x[1], reverse=True):
            text += f"- {source}: {count} 條\n"
        
        text += "\n**按分類**:\n"
        for category, count in sorted(stats['by_category'].items(), key=lambda x: x[1], reverse=True):
            text += f"- {category}: {count} 條\n"
        
        return text


if __name__ == "__main__":
    # 測試
    from core.news_aggregator import NewsAggregator
    
    aggregator = NewsAggregator()
    news = aggregator.fetch_all_news(hours=24, categories=['crypto'])
    
    builder = AIPromptBuilder()
    
    # 建構新聞 context
    news_context = builder.build_news_context(news, max_items=10)
    
    # 建構完整 prompt
    user_query = "比特幣最近的走勢如何?有什麼重大新聞影響?"
    full_prompt = builder.build_complete_prompt(user_query, news_context)
    
    print(full_prompt)
    print(f"\n\nPrompt 長度: {len(full_prompt)} 字元")
    
    # 統計
    stats = builder.get_statistics(news)
    print("\n" + builder.format_statistics(stats))
