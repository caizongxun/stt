# V1 Strategy Critical Issues Analysis
V1策略關鍵問題分析

## 現狀 (2026-02-28)

### 回測結果灣集
| 指標 | 第1次 | 第2次 | 狀態 |
|------|------|------|------|
| 總報酬 | -88.6% | -77.6% | 災難 |
| 交易次數 | 1131 | 707 | 仍過多 |
| 勝率 | 24.8% | 23.6% | 極低 |
| 盈虧比 | 0.32 | 0.30 | 災難 |
| 最大回撤 | 88.7% | 77.8% | 災難 |

## 核心問題診斷

### 1. 模型預測失敗
驗證集指標 vs 回測表現:
- 準確率55% -> 勝率24%
- 各類召回36-37% -> 實際完全失效

**根本原因**: 驗證集是訓練數據的延續,不代表真實市場。模型只學到歷史數據的模式,但市場不斷變化。

### 2. 特徵工程不足
當前特徵(25個):
- 移動平均 (SMA)
- 標準差 (STD)
- 動量 (Momentum)
- 成交量比率

**缺少**:
- 趨勢指標 (RSI, MACD, ADX)
- 支撐/壓力 (Bollinger Bands)
- 市場狀態 (趨勢/震盪)
- 多時間框架特徵

### 3. 標籤生成過於簡單
當前方法:
```python
if future_return > 1.5%: label = LONG
if future_return < -1.5%: label = SHORT
else: label = HOLD
```

**問題**:
- 沒考慮趨勢方向
- 沒考慮波動率
- 沒考慮持有時間
- 二元分類過於僵化

### 4. 回測逻輯有問題
當前逻輯:
- 收到反向信號就平倉
- 收到HOLD信號也平倉

**問題**:
- 沒有止損機制
- 沒有止盈機制
- 信號變化太頻繁導致過度交易

## 解決方案

### Option A: 改良當前V1策略 (建議)

1. **大幅改進特徵工程**
```python
# 添加技術指標
RSI(14, 28)
MACD(12, 26, 9)
Bollinger Bands(20)
ATR(14)
ADX(14)

# 添加市場狀態
trend_strength = ADX
volatility_regime = ATR / price

# 多時間框架
higher_tf_trend = SMA_1h / SMA_4h
```

2. **改進標籤生成**
```python
# 考慮趨勢和波動率
if future_return > 2% AND trend_up AND low_volatility:
    label = LONG
if future_return < -2% AND trend_down AND low_volatility:
    label = SHORT
```

3. **添加風控機制**
```python
# 回測中添加
stop_loss = 2%
take_profit = 4%
max_holding_periods = 20
```

4. **減少交易頻率**
```python
# 提高閘值到3%
label_threshold = 0.03
# 提高概率75%
probability_threshold = 0.75
# 目標: 90天<100筆交易
```

### Option B: 放棄V1,開發V2 (長期)

V2策略特點:
- 模型: LSTM/GRU (時序學習)
- 特徵: 100+技術指標
- 標籤: 連續值預測 (預測具體漲跌幅)
- 風控: 動態止損/止盈

## 立即行動 (Option A)

### 第1步: 添加技術指標
在trainer.py中添加:
```python
import ta  # 需要pip install ta

# RSI
df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)

# MACD
macd = ta.trend.MACD(df['close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
df['macd_diff'] = macd.macd_diff()

# Bollinger Bands
bb = ta.volatility.BollingerBands(df['close'])
df['bb_high'] = bb.bollinger_hband()
df['bb_low'] = bb.bollinger_lband()
df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])

# ATR
df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
```

### 第2步: 改進標籤
```python
def _generate_labels(self, df):
    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
    df['trend'] = df['close'] > df['sma_50']
    
    df['label'] = 0
    # 做多: 上漲超過3%且處於上升趨勢
    df.loc[(df['future_return'] > 0.03) & df['trend'], 'label'] = 1
    # 做空: 下跌超過3%且處於下降趨勢
    df.loc[(df['future_return'] < -0.03) & ~df['trend'], 'label'] = 2
    
    return df.dropna()
```

### 第3步: 添加風控
在backtester.py中添加:
```python
stop_loss_pct = 0.02  # 2%
take_profit_pct = 0.04  # 4%
max_holding_bars = 20

# 在模擬交易中檢查
if position != 0:
    pnl_pct = (current_price / entry_price - 1) * position
    holding_time = current_time - entry_time
    
    if pnl_pct < -stop_loss_pct:  # 止損
        should_close = True
    if pnl_pct > take_profit_pct:  # 止盈
        should_close = True
    if holding_time > max_holding_bars:  # 超時
        should_close = True
```

## 預期改善

如果完成上述改進:
- 交易次數: 700 -> 50-100
- 勝率24% -> 40-50%
- 盈虧比0.3 -> 1.2-1.5
- 總報酬: -77% -> -10%~+20%
- 最大回撤: 77% -> 20-30%

## 結論

**V1策略當前狀態: 不適合實盤**

必須先完成Option A的改進,再考慮實盤。
或者直接開發V2策略 (LSTM-based)。

---
最後更新: 2026-02-28 14:02
