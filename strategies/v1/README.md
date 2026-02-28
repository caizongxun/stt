# V1 Strategy - LightGBM Baseline

V1基礎策略 - 使用LightGBM

## 特點

- **模型**: LightGBM
- **訓練時間**: 2-5分鐘
- **GPU需求**: 不需要
- **難度**: 初級

## 使用方法

### 1. 在GUI中使用

```bash
streamlit run app.py
```

選擇V1版本,然後:
1. 選擇交易對和時間框架
2. 設定訓練參數
3. 點擊開始訓練
4. 查看回測結果

### 2. 命令行使用

```python
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

# 回測
backtester = Backtester(config)
backtest_results = backtester.run(trainer.model, df)
```

## 參數說明

### LightGBM參數

- `num_leaves`: 葉子節點數 (16-128)
- `max_depth`: 最大深度 (3-12)
- `learning_rate`: 學習率 (0.01-0.1)
- `n_estimators`: 樹數量 (50-200)

### 訓練參數

- `train_size`: 訓練集比例 (0.6-0.8)
- `lookback_periods`: 回期周期 [5, 10, 20]

## 性能指標

| 指標 | 目標 |
|------|------|
| 月報酬 | 30-50% |
| 勝率 | 55-60% |
| 盈虧比 | 1:1.5 |
| 最大回撤 | <30% |

## 注意事項

1. V1不具備時序學習能力
2. 適合快速原型驗證
3. 建議使用低槓桿(1-2x)
