# 多目標帕累托前沿探索器 ＋ 交易清單分析器

一個基於 Streamlit 的多頁應用程式：
- 「MO」頁面：用於分析回測結果，採用多目標最佳化概念，特別專注於交易策略最佳化的帕累托前沿分析。
- 「Trades」頁面：用於分析從 TradingView 匯出的 `List of Trades` 交易清單，提供進出場策略的分布、績效、風險與關聯關係視覺化。

## 功能特色

### 📊 **互動式帕累托前沿分析（MO 頁）**
- 以 2D 散點圖視覺化交易策略結果
- 自動計算並突出顯示帕累托最優解
- 支援最大化與最小化目標
- 具有懸停提示的互動式圖表

### 📈 **交易清單分析（Trades 頁）**
- 自動掃描 `ListofTrades/` 內最新的 TradingView 交易清單 CSV
- 進出場關係視覺化：
  - Sankey Diagram（進場→出場訊號流向）
  - Sunburst（Nested Pie，內圈：進場；外圈：對應出場）
- 效果與分布：
  - Stacked Bar（依 `Net P&L %` 區間，堆疊各進場訊號的交易數）
  - 熱力圖（進場×出場 的平均 `Net P&L %`）
  - 詳細統計表（總/均值 PL、交易數、PL% 均值與標準差）
- 摘要與分布：
  - 獲利摘要圖（進場/出場的總 PL、平均 PL%）
  - 小提琴圖（各進場訊號的 `Net P&L %` 分布）
  - 平均 `Net P&L %` 與 `Drawdown %` 的群組長條圖
  - 各進場訊號的統計表（mean/median/max/min/count）

### 🎯 **彈性目標選擇**
- 選擇任意兩個數值欄位作為 X 和 Y 目標
- 支援混合最佳化方向（最大化/最小化）
- 預先過濾只顯示相關的結果欄位

### 📋 **詳細帕累托分析**
- 互動式選擇帕累托前沿點
- 顯示所選點的詳細參數配置
- 匯出帕累托前沿結果為 CSV

## 安裝

### 系統需求
- Python 3.8+

## 使用方式

### 啟動應用程式
```bash
streamlit run main.py
```

應用程式為多頁架構，啟動後左側（或右上角）可切換頁面：
- MO：多目標帕累托分析
- Trades：交易清單分析

### 檔案結構
```
MO/
├── main.py                        # 主入口（MO 頁）
├── pages/
│   └── Trades Analysis.py         # 「Trades」頁（交易清單分析）
├── BackTestResults/               # MO 頁的回測結果 CSV 放這裡
│   └── your_backtest_results.csv
├── ListofTrades/                  # Trades 頁的交易清單 CSV 放這裡
│   └── your_list_of_trades.csv
└── README.md                      # 本檔案
```

## 使用指南

### 1. **準備您的資料**
- MO 頁：將回測結果 CSV 檔案放在 `BackTestResults/` 資料夾（例如 TradingView Strategy Tester 匯出）
- Trades 頁：將 TradingView 的 `List of Trades` CSV 檔案放在 `ListofTrades/` 資料夾

### 2. **選擇資料來源**
- MO 頁：自動列出 `BackTestResults/` 中的所有 `.csv`，預設選擇最近修改者
- Trades 頁：自動列出 `ListofTrades/` 中的所有 `.csv`，預設選擇最近修改者

### 3A. **（MO 頁）選擇目標**
- **目標數量**：2～5 個
- **目標欄位**：從預設清單中選擇（如 `Net profit: All`、`Max equity drawdown %`）
- **方向**：對每個目標指定最大化/最小化

### 3B. **（Trades 頁）資料欄位重點**
- 必備欄位（大小寫需一致）：
  - `Date/Time`、`Type`（包含 "Entry" / "Exit"）
  - `Trade #`、`Signal`
  - `Net P&L USDT`、`Net P&L %`、`Drawdown %`

### 4A. **（MO 頁）分析結果**
- **散點圖**：查看所有結果，帕累托前沿點以紅色突出顯示
- **帕累托前沿大小**：查看最優前沿上有多少個點
- **點選擇**：使用下拉選單檢查特定的帕累托最優配置，並可下載所選點的 CSV

### 4B. **（Trades 頁）分析結果**
- **進出場分布**：Sankey、Sunburst
- **數量分布**：依 `Net P&L %` 區間的 Stacked Bar（倒序區間）
- **獲利分析**：
  - 熱力圖：`Entry_Signal × Exit_Signal` 的平均 `Net P&L %`
  - 表格：總/平均 `PL (USDT)`、交易數、平均/標準差 `PL %`
- **獲利摘要圖**：按進場/出場聚合的總 PL 與平均 PL%
- **分布與風險**：小提琴圖（`Net P&L %`）、群組長條（平均 `Net P&L %` 與 `Drawdown %`）
- **統計表**：各進場訊號的 `Net P&L %`、`Drawdown %` 的 mean/median/max/min/count

### 5. **匯出結果**
- 下載完整的 `Pareto_Front.csv` 檔案
- 檔案名稱包含所選目標以便識別

## 了解 Pareto Front

### 什麼是 Pareto Optimality？
如果沒有其他解決方案能夠：
- 在至少一個目標上更好，且
- 在任何其他目標上都不更差

那麼這個解決方案就是 Pareto Optimal

### 範例情境
- **最大化利潤，最小化回撤**：前沿上的點代表高回報與低風險之間的最佳權衡
- **最大化夏普比率，最小化最大回撤**：最佳風險調整回報
- **最大化勝率，最小化平均損失**：最佳一致性與風險管理

### 顏色編碼
- 🔴 **紅點**：帕累托最優解（前沿）
- 🔵 **藍點**：被支配的解（內部）

## 小提示
- CSV 欄位需與程式預期一致，特別是 Trades 頁的 `Type`、`Signal`、`Trade #`。
- 若某些欄位出現空值（NaN），MO 頁會自動過濾後計算帕累托前沿。
