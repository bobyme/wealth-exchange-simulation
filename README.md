# 社會經濟模擬專案

## 目錄結構

```
.
├── wealth/                    # 財富分配模擬（Wealth Exchange Model）
│   ├── wealth_exchange.py     # 核心引擎：財富交換、稅制、UBI
│   ├── run_comparison.py      # 單次執行腳本
│   ├── run_country_comparison.py  # 四國對比（美、台、日、中）
│   └── output/                # 輸出圖表
│
├── segregation/               # 居住隔離模擬（Schelling Segregation Model）
│   ├── human_simulation.py    # 核心引擎：隔離動態、CBD 效應、仕紳化
│   ├── simulation_report.py   # 三國報告產生器（美、台、日）
│   └── output/                # 輸出圖表與 Markdown 報告
│
├── docs/                      # 文件與開發日誌
│   ├── DEV_LOG.md             # 開發討論記錄
│   └── ECON_MODELS.md         # 經濟模型說明
│
└── sim-env/                   # Python 虛擬環境
```

## 執行方式

```bash
# 啟動虛擬環境
source sim-env/bin/activate

# 居住隔離互動模擬
python segregation/human_simulation.py

# 產生三國報告（美、台、日）
python segregation/simulation_report.py

# 財富分配四國對比
python wealth/run_country_comparison.py
```

## 模型簡介

### 財富分配模擬
基於隨機財富交換模型，模擬不同稅制（所得稅、資本稅、UBI）對貧富差距（Gini 係數）的長期影響。

### 居住隔離模擬
基於 Schelling（1971）隔離模型，模擬族群/階層的空間分化，並加入 CBD 市中心效應與地價/仕紳化機制。
