# V4 Strategy - Adaptive Dual Mode System

## 核心概念

市場只有兩種狀態:
1. **盤整** (70%時間) - 區間反轉
2. **趨勢** (30%時間) - 突破跟隨

用ML識別當前狀態,自動切換策略。

## 架構

```
階段1: 市場狀態識別器 (MarketRegimeDetector)
   ↓
   ├→ 盤整模式 (ADX<25, BB窄) → 區間反轉策略
   └→ 趨勢模式 (ADX>25) → 突破跟隨策略
```

## 核心組件

### 1. MarketRegimeDetector
識別市場狀態
- ADX < 25 + BB窄 = 盤整
- ADX >= 25 = 趨勢

### 2. StructureDetector
識別支撑/壓力
- 滾動20根K棒的高低點
- 計算價格在區間位置

### 3. DualModeSignalGenerator
雙模式信號生成
- 盤整: 支撑買/壓力賣
- 趨勢: 突破追隨

### 4. AdaptiveLabelGenerator
自適應標籤生成
- 盤整: 目標=區間的50%
- 趨勢: 目標=3 ATR

## 優勢

1. **符合市場真實行為**
   - 盤整時均值回歸
   - 趨勢時動量延續

2. **策略互補**
   - 盤整: 高勝率 (55-60%)
   - 趨勢: 高盈虧比 (2-3)

3. **自動適應**
   - 不需手動切換
   - ML自動識別狀態

4. **可解釋性**
   - 每個信號有明確邏輯
   - 方便優化和調試

## 預期表現

- 正類率: 25-35%
- OOS AUC: 0.55-0.65
- 勝率: 50-55%
- 月報酬: 10-25%

## 使用

```python
from strategies.v4 import render
render()
```
