# Smart Trading Terminal - Quick Start
STT快速開始指南

## ✨ 系統介紹

STT是一個完全模塊化的加密貨幣交易策略系統。

**核心優勢:**
- 📦 每個版本獨立,互不影響
- 🖥️ 一個GUI控制所有版本
- 🔧 完整訓練/回測功能
- 📈 38個交易對, 4個時間框架

## 🚀 5分鐘快速開始

### 步驟 1: Clone 專案

```bash
git clone https://github.com/caizongxun/stt.git
cd stt
```

### 步驟 2: 安裝依賴

```bash
# 使用pip
pip install -r requirements.txt

# 或使用conda
conda create -n stt python=3.10
conda activate stt
pip install -r requirements.txt
```

### 步驟 3: 啟動GUI

```bash
streamlit run app.py
```

瀏覽器會自動打開: http://localhost:8501

### 步驟 4: 開始使用

1. **在側邊欄選擇版本**: V1/V2/V3...
2. **選擇交易對**: BTCUSDT
3. **選擇時間框架**: 15m
4. **點擊開始訓練**
5. **等待2-5分鐘**
6. **查看回測結果**

## 📚 目錄結構

```
stt/
├── app.py                  # 主程式 (僅負責頁面)
├── core/                  # 核心模組
│   ├── data_loader.py     # 數據加載器
│   ├── version_manager.py # 版本管理
│   └── gui_components.py  # GUI組件
├── strategies/            # 策略版本
│   ├── v1/                # V1策略
│   │   ├── config.py      # 配置
│   │   ├── trainer.py     # 訓練
│   │   ├── backtester.py  # 回測
│   │   └── README.md      # 說明
│   ├── v2/                # V2策略
│   └── v3/                # V3策略
├── models/                # 訓練好的模型
└── data/                  # 緩存數據
```

## 🎯 使用流程

### 1. 選擇版本

GUI左側邊欄選擇策略版本:
- **V1**: LightGBM基礎版 (推薦初學)
- **V2**: Transformer進階版
- **V3**: Kelly最優倉位版

### 2. 設定參數

**數據選擇:**
- 交易對: 38個選擇
- 時間框架: 1m/15m/1h/1d

**訓練參數:**
- 訓練集比例: 70%
- 訓練輪數: 50
- 批次大小: 64

### 3. 開始訓練

點擊「開始訓練」按鈕:
- 系統會自動下載數據
- 進行特徵工程
- 訓練模型
- 保存結果

### 4. 查看回測

切換到「回測」頁面:
- 設定回測參數
- 點擊開始回測
- 查看詳細績效

## 🔧 進階使用

### 添加新版本

1. 在`strategies/`下建立新資料夾:

```bash
mkdir -p strategies/v4
cd strategies/v4
```

2. 建立必要檔案:

```
v4/
├── __init__.py        # 必須有render()函數
├── config.py         # 配置類
├── trainer.py        # 訓練類
├── backtester.py     # 回測類
└── README.md         # 說明文檔
```

3. 在`__init__.py`中實現`render()`:

```python
import streamlit as st
from .trainer import Trainer
from .backtester import Backtester

def render():
    st.header("V4 Strategy")
    # 你的GUI逻輯
```

4. 重新啟動app.py,即可看到V4選項

### 命令行使用

```python
# 直接引用模組
from strategies.v1 import Trainer, Backtester
from strategies.v1.config import V1Config
from core.data_loader import DataLoader

# 加載數據
loader = DataLoader()
df = loader.load_klines('BTCUSDT', '15m')

# 訓練
config = V1Config()
trainer = Trainer(config)
results = trainer.train(df)

print(f"訓練完成: {results}")
```

## 📈 數據說明

### HuggingFace數據集

**Dataset ID**: `zongowo111/v2-crypto-ohlcv-data`

**交易對 (38個)**:
```
BTCUSDT, ETHUSDT, BNBUSDT, ADAUSDT, SOLUSDT,
AVAXUSDT, DOTUSDT, MATICUSDT, LINKUSDT, UNIUSDT,
...
```

**時間框架 (4個)**:
- `1m`: 1分鐘 (約2年數據)
- `15m`: 15分鐘
- `1h`: 1小時
- `1d`: 日線

**數據欄位**:
```
open_time, open, high, low, close, volume,
close_time, quote_asset_volume, number_of_trades,
taker_buy_base_asset_volume, taker_buy_quote_asset_volume
```

### 本地緩存

首次下載後,數據會緩存在`data/cache/`:
- 後續使用會直接讀取緩存
- 無需重複下載
- 加快訓練速度

## ❓ 常見問題

### Q: 如何添加新策略?

A: 複製`strategies/v1/`到`strategies/v4/`,修改內容即可。

### Q: 如何使用自己的數據?

A: 修改`core/data_loader.py`,實現自己的加載函數。

### Q: 如何部署到服務器?

A: 
```bash
# 使用streamlit cloud
streamlit deploy app.py

# 或使用docker
docker build -t stt .
docker run -p 8501:8501 stt
```

### Q: GPU訓練如何配置?

A: V1不需GPU。V2/V3需要PyTorch:
```bash
pip install torch torchvision torchaudio
```

## 📄 相關文檔

- [V1策略說明](strategies/v1/README.md)
- [HuggingFace數據集](https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data)
- [GitHub Repo](https://github.com/caizongxun/stt)

## ⚠️ 風險聲明

1. 此系統僅供學習研究
2. 加密貨幣交易高風險
3. 歷史表現≠未來結果
4. 實盤前充分測試

## 👍 貢獻

歡迎PR和Issue!

---

**開發者**: caizongxun  
**更新**: 2026-02-28  
**版本**: 1.0.0
