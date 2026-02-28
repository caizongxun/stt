# Smart Trading Terminal (STT)

🚀 模塊化加密貨幣交易策略系統

## 🎯 特點

- 📦 **模塊化架構**: 每個版本獨立,易於維護
- 🖥️ **統一GUI**: 一個界面控制所有版本
- 🔧 **完整功能**: 訓練/回測/參數調整
- 📈 **HuggingFace數據**: 38個交易對, 4個時間框架
- ⚡ **快速切換**: 不同策略一鍵切換

## 🏛️ 架構設計

```
stt/
├── app.py                    # 主界面 (僅負責頁面與呼叫)
├── core/                    # 核心模組
│   ├── __init__.py
│   ├── data_loader.py       # 統一數據加載器
│   ├── version_manager.py   # 版本管理器
│   └── gui_components.py    # GUI共用組件
├── strategies/              # 策略版本
│   ├── v1/                  # 第一版策略
│   │   ├── __init__.py
│   │   ├── config.py        # 配置檔
│   │   ├── trainer.py       # 訓練模組
│   │   ├── backtester.py    # 回測模組
│   │   ├── strategy.py      # 策略逻輯
│   │   └── README.md        # 版本說明
│   ├── v2/                  # 第二版策略
│   │   └── ...
│   └── v3/                  # 第三版策略
│       └── ...
├── models/                  # 訓練完成的模型
├── data/                    # 本地緩存數據
├── requirements.txt         # 依賴套件
└── README.md                # 專案說明
```

## ⚡ 快速開始

### 1. 安裝依賴

```bash
git clone https://github.com/caizongxun/stt.git
cd stt
pip install -r requirements.txt
```

### 2. 啟動GUI

```bash
streamlit run app.py
```

### 3. 使用流程

1. 📋 **選擇版本**: 在側邊欄選擇策略版本 (v1/v2/v3...)
2. 🔧 **設定參數**: 交易對/時間框架/模型參數
3. 🚀 **開始訓練**: 點擊訓練按鈕
4. 📊 **查看回測**: 切換到回測頁面

## 📈 數據源

**HuggingFace Dataset**: `zongowo111/v2-crypto-ohlcv-data`

- **38個交易對**: BTCUSDT, ETHUSDT, BNBUSDT, ...
- **4個時間框架**: 1m, 15m, 1h, 1d
- **約150萬筆數據**: 全自動下載緩存

## 📦 版本策略

### V1 - 基礎策略
- 模型: XGBoost/LightGBM
- 特點: 簡單快速
- 適用: 初學者

### V2 - 進階策略  
- 模型: Transformer
- 特點: 時序學習
- 適用: 中階交易者

### V3 - Kelly策略
- 模型: LSTM/GRU + Kelly
- 特點: 最優倉位
- 適用: 進階交易者

### V6 - 反轉預測 (新增)
- 模型: LightGBM 雙向分類
- 特點: 結合BB位置 + RSI 尋找極端反轉點
- 適用: 需要防禦趨勢行情持續虧損時的反轉捕捉

## ⚠️ 風險聲明

1. 此系統僅供學習和研究使用
2. 加密貨幣交易具有高風險
3. 歷史表現不代表未來結果
4. 實盤前請充分測試

---

**開發者**: caizongxun  
**更新日期**: 2026-02-28
