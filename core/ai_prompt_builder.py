"""AI Prompt Builder - Enhanced with WorldMonitor Integration

建構給 AI 的完整 context,深度整合新聞資訊
"""

from typing import List, Dict, Optional
from datetime import datetime
import textwrap


class AIPromptBuilder:
    """AI Prompt 建構器 - WorldMonitor 增強版"""
    
    @staticmethod
    def build_news_context(
        news_list: List[Dict],
        max_items: int = 20,
        max_content_length: int = 2000,
        include_full_content: bool = True,
        highlight_alerts: bool = True
    ) -> str:
        """建構新聞 context (WorldMonitor 風格)
        
        Args:
            news_list: 新聞列表
            max_items: 最多包含幾條新聞
            max_content_length: 每條新聞內容最大長度
            include_full_content: 是否包含完整內容
            highlight_alerts: 是否突顯高風險新聞
            
        Returns:
            格式化的新聞 context
        """
        if not news_list:
            return "# 📰 新聞資訊\n\n目前沒有最新新聞。\n\n"
        
        # 分離警報新聞
        alerts = [item for item in news_list if item.get('is_alert', False)][:5]
        regular_news = [item for item in news_list if not item.get('is_alert', False)]
        
        context = "# 🌍 WorldMonitor - 全球新聞與市場動態\n\n"
        context += f"**Context Generation Time**: {datetime.now().isoformat()}\n"
        context += f"**Total News Items**: {len(news_list[:max_items])}\n"
        
        if alerts and highlight_alerts:
            context += f"**⚠️ High-Priority Alerts**: {len(alerts)}\n"
        
        context += "\n" + "=" * 100 + "\n\n"
        
        # === 高風險警報區 ===
        if alerts and highlight_alerts:
            context += "## 🚨 HIGH-PRIORITY ALERTS\n\n"
            context += "These items have been flagged as critical or high-risk. Pay special attention.\n\n"
            
            for idx, item in enumerate(alerts, 1):
                threat = item.get('threat', {})
                level = threat.get('level', 'unknown').upper()
                keyword = threat.get('keyword', '')
                
                context += f"### ⚠️ Alert {idx}: [{level}] {item['title']}\n\n"
                context += f"- **Source**: {item['source']}\n"
                context += f"- **Category**: {item['category']}\n"
                context += f"- **Published**: {item['published'].strftime('%Y-%m-%d %H:%M UTC')}\n"
                context += f"- **Threat Level**: {level}\n"
                
                if keyword:
                    context += f"- **Trigger Keyword**: {keyword}\n"
                
                context += f"- **URL**: {item['link']}\n\n"
                
                # 內容
                content = item.get('full_content', item.get('summary', ''))[:max_content_length]
                if content:
                    context += f"**Content**:\n{content}\n"
                    if len(item.get('full_content', '')) > max_content_length:
                        context += "\n[Content truncated for brevity...]\n"
                
                context += "\n" + "-" * 100 + "\n\n"
        
        # === 一般新聞區 ===
        context += "## 📰 Regular News & Market Updates\n\n"
        
        remaining_slots = max_items - len(alerts) if highlight_alerts else max_items
        
        for idx, item in enumerate(regular_news[:remaining_slots], 1):
            context += f"### News {idx}: {item['title']}\n\n"
            
            # 元資料
            context += f"- **Source**: {item['source']}\n"
            context += f"- **Category**: {item['category']}\n"
            context += f"- **Published**: {item['published'].strftime('%Y-%m-%d %H:%M')}\n"
            context += f"- **URL**: {item['link']}\n"
            
            if item.get('image_url'):
                context += f"- **Image**: {item['image_url']}\n"
            
            # 威脅等級 (如果有)
            if 'threat' in item:
                threat = item['threat']
                if threat.get('level') not in ['low', 'unknown']:
                    context += f"- **Risk Level**: {threat.get('level', 'low')}\n"
            
            context += "\n"
            
            # 內容
            if include_full_content and item.get('full_content'):
                content = item['full_content'][:max_content_length]
                if len(item['full_content']) > max_content_length:
                    content += "\n\n[Content truncated...]"
                context += f"**Full Content**:\n{content}\n\n"
            else:
                summary = item.get('summary', '')[:800]
                if len(item.get('summary', '')) > 800:
                    summary += "..."
                if summary:
                    context += f"**Summary**: {summary}\n\n"
            
            context += "-" * 100 + "\n\n"
        
        # === 分析指引 ===
        context += "\n" + "=" * 100 + "\n\n"
        context += "## 📊 AI Analysis Guidelines\n\n"
        context += textwrap.dedent("""
        **Context Usage Instructions**:
        1. The above news items are machine-collected from various sources (RSS feeds)
        2. Treat this information as **unverified signals** - always cross-check critical claims
        3. High-priority alerts have been automatically flagged based on keywords
        4. Consider the publication timestamp when assessing relevance
        5. Use this context to inform your analysis, but do NOT hallucinate details not present
        
        **Key Focus Areas for Trading Analysis**:
        - Regulatory changes and policy announcements
        - Major exchange activities and liquidity events
        - Technical developments and protocol upgrades
        - Market sentiment shifts and social indicators
        - Macroeconomic factors affecting crypto markets
        - Security incidents and risk events
        
        **Response Guidelines**:
        - Cite specific news items when making claims
        - Distinguish between confirmed facts and speculation
        - Acknowledge uncertainty when data is incomplete
        - Provide balanced analysis considering multiple perspectives
        """).strip()
        
        context += "\n\n" + "=" * 100 + "\n\n"
        
        return context
    
    @staticmethod
    def build_complete_prompt(
        user_query: str,
        news_context: str,
        market_data: Optional[Dict] = None,
        system_instruction: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> str:
        """建構完整的 AI prompt (適用於 DeepSeek / GPT 等模型)
        
        Args:
            user_query: 用戶問題
            news_context: 新聞 context (from build_news_context)
            market_data: 市場數據 (價格、指標等)
            system_instruction: 自訂系統指令
            additional_context: 額外 context
            
        Returns:
            完整 prompt (可直接送給 AI 模型)
        """
        prompt = ""
        
        # === 系統指令 ===
        if system_instruction:
            prompt += f"{system_instruction}\n\n"
        else:
            prompt += textwrap.dedent("""
            # SYSTEM ROLE
            
            You are an expert cryptocurrency trading analyst and market researcher.
            
            ## Your Expertise:
            - Advanced technical analysis (indicators, patterns, multi-timeframe)
            - Fundamental analysis (on-chain metrics, project fundamentals)
            - Market sentiment analysis (news impact, social indicators)
            - Risk management and position sizing
            - Trading strategy development
            - Macro-economic factors affecting crypto
            
            ## Response Principles:
            1. **Evidence-Based**: Ground all analysis in data and verifiable facts
            2. **Context-Aware**: Integrate latest news and market conditions
            3. **Multi-Dimensional**: Consider technical, fundamental, and sentiment factors
            4. **Risk-Conscious**: Always highlight potential risks and uncertainties
            5. **Clear Communication**: Use Traditional Chinese (繁體中文) unless otherwise specified
            6. **Actionable**: Provide concrete insights and recommendations when appropriate
            
            ## Critical Guidelines:
            - DO cite specific news items or data points when making claims
            - DO NOT fabricate statistics or make up information not in context
            - DO acknowledge when information is uncertain or incomplete
            - DO provide multiple perspectives when appropriate
            - DO warn about high-risk scenarios explicitly
            """).strip()
        
        prompt += "\n\n" + "=" * 100 + "\n\n"
        
        # === 市場數據 (如果有) ===
        if market_data:
            prompt += "## 📈 Current Market Data\n\n"
            
            if 'price' in market_data:
                prompt += f"- **Current Price**: ${market_data['price']:,.2f}\n"
            if 'change_24h' in market_data:
                prompt += f"- **24h Change**: {market_data['change_24h']:+.2f}%\n"
            if 'volume_24h' in market_data:
                prompt += f"- **24h Volume**: ${market_data['volume_24h']:,.0f}\n"
            if 'timestamp' in market_data:
                prompt += f"- **Data Timestamp**: {market_data['timestamp']}\n"
            
            prompt += "\n" + "=" * 100 + "\n\n"
        
        # === 新聞 Context ===
        prompt += news_context
        
        # === 額外 Context ===
        if additional_context:
            prompt += "## 📎 Additional Context\n\n"
            prompt += additional_context + "\n\n"
            prompt += "=" * 100 + "\n\n"
        
        # === 用戶問題 ===
        prompt += "## 💬 User Query\n\n"
        prompt += f"{user_query}\n\n"
        prompt += "=" * 100 + "\n\n"
        
        # === 回答提示 ===
        prompt += "## 📝 Your Task\n\n"
        prompt += textwrap.dedent("""
        Provide a comprehensive analysis addressing the user's query.
        
        **Structure your response**:
        1. **Direct Answer**: Start with a clear, concise answer to the main question
        2. **Supporting Analysis**: Provide detailed reasoning and evidence
        3. **News Context**: Reference relevant news items from the context above
        4. **Risk Assessment**: Highlight potential risks and uncertainties
        5. **Actionable Insights**: Conclude with practical recommendations if appropriate
        
        **Remember**: 
        - Use Traditional Chinese (繁體中文)
        - Cite specific news items when relevant
        - Be transparent about confidence levels
        - Distinguish between facts and analysis
        """).strip()
        
        prompt += "\n\n" + "=" * 100 + "\n"
        
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
                'alerts': 0,
                'by_source': {},
                'by_category': {},
                'by_threat_level': {},
                'time_range': None
            }
        
        stats = {
            'total': len(news_list),
            'alerts': sum(1 for item in news_list if item.get('is_alert', False)),
            'by_source': {},
            'by_category': {},
            'by_threat_level': {},
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
            
            # 按威脅等級統計
            if 'threat' in item:
                level = item['threat'].get('level', 'unknown')
                stats['by_threat_level'][level] = stats['by_threat_level'].get(level, 0) + 1
        
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
        
        text = "# 📊 新聞統計\n\n"
        text += f"**總數**: {stats['total']} 條\n"
        text += f"**警報**: {stats['alerts']} 條\n\n"
        
        # 時間範圍
        if stats['time_range']:
            earliest = stats['time_range']['earliest'].strftime('%Y-%m-%d %H:%M')
            latest = stats['time_range']['latest'].strftime('%Y-%m-%d %H:%M')
            text += f"**時間範圍**: {earliest} ~ {latest}\n\n"
        
        # 按威脅等級
        if stats['by_threat_level']:
            text += "**威脅等級分布**:\n"
            for level in ['critical', 'high', 'medium', 'low', 'unknown']:
                if level in stats['by_threat_level']:
                    count = stats['by_threat_level'][level]
                    text += f"- {level.upper()}: {count} 條\n"
            text += "\n"
        
        # 按來源
        text += "**按來源**:\n"
        for source, count in sorted(stats['by_source'].items(), key=lambda x: x[1], reverse=True)[:10]:
            text += f"- {source}: {count} 條\n"
        
        text += "\n**按分類**:\n"
        for category, count in sorted(stats['by_category'].items(), key=lambda x: x[1], reverse=True):
            text += f"- {category}: {count} 條\n"
        
        return text


if __name__ == "__main__":
    # === 測試範例 ===
    print("測試 AI Prompt Builder (WorldMonitor Enhanced)\n")
    print("=" * 80)
    
    try:
        from core.news_aggregator import NewsAggregator
        
        # 初始化
        aggregator = NewsAggregator()
        builder = AIPromptBuilder()
        
        # 抓取新聞
        print("\n📰 抓取最新加密貨幣新聞...\n")
        news = aggregator.fetch_all_news(
            hours=24,
            include_content=False,  # 測試時不爬完整內容
            categories=['crypto']
        )
        
        if news:
            # 建構新聞 context
            news_context = builder.build_news_context(
                news,
                max_items=15,
                include_full_content=False,
                highlight_alerts=True
            )
            
            # 建構完整 prompt
            user_query = "比特幣最近的走勢如何?有什麼重大新聞影響市場?"
            
            full_prompt = builder.build_complete_prompt(
                user_query=user_query,
                news_context=news_context,
                market_data={
                    'price': 65432.10,
                    'change_24h': +2.34,
                    'volume_24h': 28500000000,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # 輸出結果
            print("\n" + "=" * 80)
            print("✅ Prompt 建構完成!")
            print("=" * 80)
            print(f"\nPrompt 總長度: {len(full_prompt):,} 字元")
            print(f"Token 估算: ~{len(full_prompt) // 4:,} tokens")
            
            # 統計
            stats = builder.get_statistics(news)
            print("\n" + builder.format_statistics(stats))
            
            # 存檔供檢視
            output_file = "test_prompt.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_prompt)
            print(f"\n💾 完整 prompt 已存檔至: {output_file}")
            
        else:
            print("\n❌ 沒有抓取到新聞")
    
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
