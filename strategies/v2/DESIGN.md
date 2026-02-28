# V2 Strategy Design - BB Reversal System
V2策略設計 - BB反轉系統

## 核心概念

**不預測方向,而是預測反轉成功率**

當K棒觸碰BB通道時,模型判斷是否會有效反轉,並輸出概率。

## 系統架構

```
[市場數據] 
    ↓
[特徵工程] 
    ↓
[BB觸碰檢測] 
    ↓
[模型預測] -> 反轉概率 (0-1)
    ↓
[進場策略] -> 馬丁/分批
    ↓
[風控管理] -> ATR動態止盈止損
```

## Part 1: 標籤生成設計

### 1.1 BB觸碰定義

```python
# 觸碰上軌
touch_upper = (high >= bb_upper) & (close < bb_upper)

# 觸碰下軌
touch_lower = (low <= bb_lower) & (close > bb_lower)

# 觸碰力道
touch_strength = abs(high - bb_upper) / atr  # 或 abs(low - bb_lower) / atr
```

### 1.2 有效反轉定義

**上軌觸碰後的有效反轉** (下跌):
```python
# 條件1: 必須有明顯下跌
future_low = df['low'].rolling(10).min().shift(-10)
has_drop = (future_low < close - 1.5 * atr)

# 條件2: 反轉後不能突破上軌
no_breakout = (df['high'].rolling(10).max().shift(-10) < bb_upper * 1.01)

# 條件3: 有止盈機會 (下跌至少到中軌)
reach_middle = (future_low <= bb_middle)

# 有效反轉
label_short = touch_upper & has_drop & no_breakout & reach_middle
```

**下軌觸碰後的有效反轉** (上漲):
```python
future_high = df['high'].rolling(10).max().shift(-10)
has_rise = (future_high > close + 1.5 * atr)
no_breakout = (df['low'].rolling(10).min().shift(-10) > bb_lower * 0.99)
reach_middle = (future_high >= bb_middle)

label_long = touch_lower & has_rise & no_breakout & reach_middle
```

### 1.3 止盈止損設計

```python
# 動態ATR止盈止損
stop_loss = 2.0 * atr      # 2倍ATR
take_profit = 3.0 * atr    # 3倍ATR

# 或使用BB通道寬度
bb_width = bb_upper - bb_lower
stop_loss = 0.5 * bb_width
take_profit = 1.0 * bb_width
```

## Part 2: 特徵工程

### 2.1 BB相關特徵

```python
# 基礎BB
bb_position = (close - bb_lower) / (bb_upper - bb_lower)
bb_width = (bb_upper - bb_lower) / bb_middle
bb_squeeze = bb_width < bb_width.rolling(20).mean() * 0.8

# BB斜率
bb_upper_slope = (bb_upper - bb_upper.shift(5)) / bb_upper.shift(5)
bb_lower_slope = (bb_lower - bb_lower.shift(5)) / bb_lower.shift(5)

# 距離BB通道的距離
dist_to_upper = (bb_upper - close) / atr
dist_to_lower = (close - bb_lower) / atr
```

### 2.2 市場環境特徵

```python
# 趨勢強度
adx = ta.trend.adx(high, low, close, window=14)
trend_direction = 1 if sma_20 > sma_50 else -1

# 波動率狀態
atr_ratio = atr / close
volatility_regime = 'high' if atr_ratio > atr_ratio.rolling(50).mean() * 1.2 else 'normal'

# 成交量
volume_spike = volume > volume.rolling(20).mean() * 1.5
```

### 2.3 歷史反轉統計

```python
# 過去20次觸碰的成功率
recent_touch_upper = touch_upper.rolling(100).sum()
recent_success_upper = label_short.rolling(100).sum()
success_rate_upper = recent_success_upper / recent_touch_upper

# 當前時間段的成功率 (亞洲/歐洲/美洲時段)
hour = df['open_time'].dt.hour
session_success_rate = df.groupby(hour)['label'].transform('mean')
```

## Part 3: 模型設計

### 3.1 模型選擇

**Option A: LightGBM** (快速)
```python
model = lgb.LGBMClassifier(
    objective='binary',
    num_leaves=20,
    max_depth=4,
    learning_rate=0.03
)
```

**Option B: LSTM** (時序)
```python
model = Sequential([
    LSTM(64, input_shape=(lookback, n_features), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
```

### 3.2 輸入/輸出

```python
# 輸入: 只在觸碰BB時間點
X = features[touch_upper | touch_lower]

# 輸出: 二元分類
y = label_short | label_long  # 0/1

# 或三元分類
y = [0: 無效反轉, 1: 做多, 2: 做空]
```

## Part 4: 進場策略

### 4.1 策略A: 馬丁格爾 (激進)

```python
class MartingaleEntry:
    def __init__(self):
        self.base_size = 0.1  # 10%資金
        self.max_levels = 3
        self.multiplier = 2.0
        self.grid_distance = 1.0 * atr
    
    def execute(self, signal_price, direction):
        levels = []
        for i in range(self.max_levels):
            if direction == 'LONG':
                entry_price = signal_price - i * self.grid_distance
            else:
                entry_price = signal_price + i * self.grid_distance
            
            position_size = self.base_size * (self.multiplier ** i)
            levels.append({
                'price': entry_price,
                'size': position_size
            })
        return levels

# 範例:
# 信號: 做多 @ 10000
# Level 1: 10000, 10%
# Level 2: 9950 (-1 ATR), 20%
# Level 3: 9900 (-2 ATR), 40%
# 平均成本: 9928.57
```

### 4.2 策略B: 分批進場 (穩健)

```python
class BatchEntry:
    def __init__(self):
        self.total_size = 0.3  # 30%資金
        self.num_batches = 3
        self.wait_bars = 2  # 每2根K棒加倉一次
    
    def execute(self, signal_price, direction, current_bar):
        size_per_batch = self.total_size / self.num_batches
        entries = []
        
        for i in range(self.num_batches):
            entry_bar = current_bar + i * self.wait_bars
            entries.append({
                'bar': entry_bar,
                'size': size_per_batch,
                'condition': 'price_still_favorable'  # 價格仍在合理範圍
            })
        return entries

# 範例:
# 信號: 做多 @ bar 100
# Batch 1: bar 100, 10%
# Batch 2: bar 102, 10% (如果價格未上漨過多)
# Batch 3: bar 104, 10% (如果價格未上漨過多)
```

### 4.3 策略C: 混合型 (推薦)

```python
class HybridEntry:
    def __init__(self):
        self.initial_size = 0.15  # 15%立即進場
        self.add_on_size = 0.1    # 10%加倉
        self.max_add_ons = 2
        self.add_condition = 'price_goes_against'  # 價格逆向才加倉
    
    def execute(self, signal_price, direction):
        # 立即進場
        positions = [{'type': 'initial', 'size': self.initial_size}]
        
        # 加倉條件
        for i in range(self.max_add_ons):
            if direction == 'LONG':
                trigger_price = signal_price * (1 - 0.005 * (i+1))  # -0.5%
            else:
                trigger_price = signal_price * (1 + 0.005 * (i+1))  # +0.5%
            
            positions.append({
                'type': 'add_on',
                'trigger': trigger_price,
                'size': self.add_on_size
            })
        return positions

# 範例:
# 信號: 做多 @ 10000
# 立即: 10000, 15%
# 加倓1: 9950 (-0.5%), 10% (觸發才執行)
# 加倓2: 9900 (-1.0%), 10% (觸發才執行)
```

## Part 5: 風控管理

### 5.1 動態止盈止損

```python
class DynamicStopLoss:
    def __init__(self, atr):
        self.atr = atr
        self.initial_sl = 2.0 * atr
        self.initial_tp = 3.0 * atr
    
    def update(self, entry_price, current_price, current_atr, direction):
        # 止損隨ATR調整
        current_sl = 2.0 * current_atr
        
        # 移動止損 (盈利後)
        if direction == 'LONG':
            profit = current_price - entry_price
            if profit > 1.0 * self.atr:
                # 盈利超過1倍ATR,移動止損到保本
                sl_price = entry_price
            elif profit > 2.0 * self.atr:
                # 盈利超過2倍ATR,移動止損到1倍ATR
                sl_price = entry_price + 1.0 * self.atr
            else:
                sl_price = entry_price - current_sl
        
        return sl_price
```

### 5.2 總風險控制

```python
# 單筆交易最大風險
max_risk_per_trade = 0.02  # 2%

# 同時最多持倉數
max_concurrent_positions = 2

# 每日最大交易次數
max_trades_per_day = 5

# 最大回撤限制
if current_drawdown > 0.20:  # 20%
    stop_trading = True
```

## Part 6: 回測流程

```python
# 偽代碼
for bar in df:
    # 1. 檢查BB觸碰
    if touch_bb:
        # 2. 模型預測
        prob = model.predict_proba(features)
        
        if prob > 0.7:  # 高概率信號
            # 3. 執行進場策略
            positions = entry_strategy.execute(price, direction)
            
            # 4. 設置止盈止損
            sl = entry_price - 2.0 * atr
            tp = entry_price + 3.0 * atr
    
    # 5. 管理持倉
    if has_position:
        # 更新止損
        sl = dynamic_sl.update(entry, current, atr, direction)
        
        # 檢查出場
        if current_price <= sl or current_price >= tp:
            close_position()
```

## 預期表現

如果設計正確:
- **勝率60-70%** (因為只交易高概率信號)
- **盈虧比1.5-2.0** (因為TP > SL)
- **月報酬10-20%** (低頻交易)
- **最大回撤10-15%** (因為風控嚴格)
- **月交易次數10-30筆** (只交易BB觸碰)

---

下一步: 開始實現 V2_predictor 模組
