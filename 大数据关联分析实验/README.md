# 大数据关联分析实验

## 项目简介
本项目基于超市购物篮数据，采用 Apriori 和 FP-growth 两种经典关联规则挖掘算法，分别实现了手写版与库函数版，并对比其性能和结果。项目涵盖数据探索、预处理、频繁项集与关联规则挖掘、可视化分析及业务建议。

## 目录结构
```
├── data/               # 数据集目录
│   └── BreadBasket_DMS.csv  # 原始购物篮数据
├── output/             # 输出结果目录
│   └── results.csv         # 实验输出示例
├── src/                # 源代码目录
│   ├── market_basket_analysis.py  # 主实验流程（含可视化和业务分析）
│   ├── apriori_handwritten.py     # Apriori 手写实现
│   ├── apriori_library.py         # Apriori 库函数实现（mlxtend）
│   ├── fpgrowth_handwritten.py    # FP-growth 手写实现
│   ├── fpgrowth_library.py        # FP-growth 库函数实现（mlxtend）
│   └── test_all_algorithms.py     # 四种算法对比测试脚本
├── requirements.txt     # Python依赖包
└── README.md            # 项目说明文档
```

## 依赖环境
- Python 3.x
- pandas
- mlxtend
- matplotlib

安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法
1. 数据分析与可视化（推荐）：
   ```bash
   python src/market_basket_analysis.py
   ```
   - 步骤包括：数据探索、清洗、热销商品与交易量分析、Apriori/FP-growth 挖掘、可视化和业务建议。

2. 四种算法对比测试（小样本）：
   ```bash
   python src/test_all_algorithms.py
   ```
   - 对比 Apriori/FP-growth 的手写与库函数实现，输出频繁项集与关联规则。

## 主要功能说明
- `market_basket_analysis.py`：
  - 数据探索与可视化
  - 数据清洗与预处理
  - 频繁项集与关联规则分析（Apriori、FP-growth，均基于 mlxtend 库）
  - 业务建议输出
- `apriori_handwritten.py`、`fpgrowth_handwritten.py`：
  - 纯 Python 手写算法实现
- `apriori_library.py`、`fpgrowth_library.py`：
  - 基于 mlxtend 的算法封装与用法示例
- `test_all_algorithms.py`：
  - 四种实现的对比测试，适合理解算法异同

## 数据说明
- 数据集：`data/BreadBasket_DMS.csv`，包含超市交易流水，字段有 Date、Time、Transaction、Item。
- 输出结果：部分分析结果保存在 `output/results.csv`。

## 实验结果与业务解读（示例）
- 热销商品主要为咖啡、面包等，可重点备货。
- 频繁项集和高置信度规则显示部分商品常被一起购买（如 Coffee 与 Bread），可考虑捆绑销售。
- 周末交易量显著高于工作日，可在周末增加促销活动。
- FP-growth 与 Apriori 结果高度一致，FP-growth 更适合大数据场景。

## 致谢
感谢 mlxtend 库提供高效的关联规则挖掘工具。
