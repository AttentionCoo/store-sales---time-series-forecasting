# 📈 Store Sales - Time Series Forecasting (LightGBM 方案)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-Latest-orange?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-Dataframe-blueviolet?style=flat-square&logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=flat-square&logo=scikit-learn)
![Competition](https://img.shields.io/badge/Kaggle-Store%20Sales-cyan?style=flat-square&logo=kaggle)

<p align="center">
  <b>🚀 基于传统机器学习与高阶特征工程的 Corporación Favorita 商店销售额预测系统</b>
  <br />
  打破深度学习神话 · 聚焦纯表格时序特征挖掘 · 极致轻量化的 GBDT 提速方案
</p>

</div>

---

## 📖 项目简介

本项目用于解决 Kaggle 经典时间序列经典比赛 **[Store Sales - Time Series Forecasting](https://www.kaggle.com/c/store-sales-time-series-forecasting)**。

与重资源的深度学习（如 LSTM, Transformer）路线不同，本项目完全基于**传统机器学习（Machine Learning Pipeline）**，利用强大的分布式梯度提升树模型 **LightGBM**，配合极其精细的**时间序列特征工程**，在保持极快训练速度的同时，实现对商店未来全品类商品销售额的精准预测。

---

## 📂 项目结构说明 (Repository Structure)

仓库中包含多个 Jupyter Notebook，分别承担了从数据探索（EDA）到最终融合的不同实验分支：

* **`kaggle_test.ipynb`**：**核心主线脚本**。包含了完整的多源数据合并流程、高阶特征提取、LightGBM 调参以及生成最终提交文件的全生命周期代码。
* **`test.ipynb` / `origin.ipynb`**：数据探索性分析（EDA）与基线模型（Baseline）构建。
* **`xsyhhh.ipynb`**：用于模型交叉验证（CV）、特征重要性筛选及实验对比的分支实验。

---

## 🛠 核心机器学习流水线 (ML Pipeline)

### 1. 多源数据融合与预处理
* **多维拼表**：以 `train.csv` / `test.csv` 为骨架，动态级联外键数据，包括商店元数据 (`stores.csv`)、交易历史 (`transactions.csv`)、国家与地方节假日 (`holidays_events.csv`) 以及宏观每日油价 (`oil.csv`)。
* **时序缺失值对齐**：针对油价等在周末或节假日不更新的断档数据，采用基于时间轴的连续前向填充 (`ffill`) 与后向填充 (`bfill`) 算法进行平滑处理。

### 2. 特征工程构建 (Feature Engineering —— 本项目灵魂)
为了让 GBDT 树模型能够理解时间本身的结构，构建了四大类、数百维度的特异性特征：
* **基础时序拆解**：提取 `Year`, `Month`, `Day`, `DayOfWeek`, `DayOfYear`, `WeekOfYear`, `Quarter` 以及逻辑标量 `is_weekend`, `is_month_start`, `is_month_end`。
* **滞后特征 (Lag Features)**：构造销售额在历史 $t-1, t-7, t-14, t-28$ 天的滞后值，强行赋予非时序模型捕获自相关性（Autocorrelation）与周期性的能力。
* **滚动统计特征 (Rolling Windows)**：基于 7 天、14 天、28 天窗口计算移动平均（Rolling Mean）、移动标准差，从而捕捉中期消费趋势。
* **多维异构衍生**：计算交叉衍生项，如单次交易平均销售额 (`sales_per_transaction`)、油价多日滑动均值等。
* **高基数分类变量编码**：使用 `LabelEncoder` 对商品大类 (`family`)、城市 (`city`)、州 (`state`)、假日类型进行数值矩阵化映射。

### 3. 模型训练与时序验证
* **目标值稳定化**：对偏态严重的销售额目标值 (`sales`) 进行 `log1p` 对数变换以稳定方差，缓解极端秒杀或大促带来的方差爆炸，模型预测后再通过 `expm1` 逆变换还原。
* **防止未来穿越 (Time Series Split)**：摒弃传统的随机 K-Fold，采用基于时间切片的严苛划分法（以 2017-06-15 / 2017-07-01 为界划分验证集），确保绝不用“未来数据”训练当前模型。
* **评估指标**：严格对齐官方榜单，使用 **RMSLE (均方根对数误差)** 进行本地线下评估。

---

## 🚀 快速启动 (Quick Start)

### 1. 数据准备
前往 Kaggle 下载比赛数据集，在项目根目录下创建 `data/` 文件夹并将解压后的 `.csv` 文件放置其中：
```text
📂 Store-Sales-Forecasting
├── 📂 data
│   ├── train.csv
│   ├── test.csv
│   ├── oil.csv
│   └── ...
└── kaggle_test.ipynb
