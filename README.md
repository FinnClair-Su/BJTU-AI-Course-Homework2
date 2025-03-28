# 机器学习分类与回归模型实践项目 🤖

<div align="center">

![Python](https://img.shields.io/badge/Python-3.6%2B-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-F7931E?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/pandas-latest-150458?style=flat-square&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-latest-013243?style=flat-square&logo=numpy)
![XGBoost](https://img.shields.io/badge/XGBoost-optional-006ACC?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

<p align="center">
  <b>一个用于探索和比较多种机器学习算法的完整项目框架</b>
</p>

这个项目实现了一套完整的机器学习工作流程，包括数据加载、预处理、模型训练、评估和可视化。项目涵盖两个具体应用场景：乳腺癌数据分类和房价数据分析（回归与分类）。通过这个项目，你可以学习如何应用多种机器学习算法解决实际问题，并比较不同算法的性能。

## 📋 目录

- [项目结构](#-项目结构)
- [功能特点](#-功能特点)
- [环境要求](#-环境要求)
- [安装指南](#-安装指南)
- [使用方法](#-使用方法)
- [输出结果](#-输出结果)
- [扩展项目](#-扩展项目)
- [常见问题](#-常见问题)
- [许可证](#-许可证)

## 📁 项目结构

```
ml_project/
├── src/
│   ├── data/
│   │   ├── init.py
│   │   ├── breast_cancer_loader.py  # 乳腺癌数据加载
│   │   └── real_estate_loader.py    # 房价数据加载
│   ├── models/
│   │   ├── init.py
│   │   ├── classification/
│   │   │   ├── init.py
│   │   │   ├── decision_tree.py     # 决策树实现
│   │   │   ├── knn.py               # KNN实现
│   │   │   ├── linear_model.py      # 线性模型实现
│   │   │   ├── svm.py               # SVM实现
│   │   │   └── price_classifiers.py # 房价分类模型
│   │   └── regression/
│   │       ├── init.py
│   │       └── price_regression.py  # 房价回归模型
│   ├── utils/
│   │   ├── init.py
│   │   ├── evaluation.py            # 评估指标
│   │   └── visualization.py         # 绘图函数
│   ├── init.py
│   └── main.py                      # 主程序入口
├── plots/                           # 存放图表
└── data.xlsx                        # 房价数据
```

## ✨ 功能特点

### 🔍 乳腺癌数据分类

使用scikit-learn内置的乳腺癌数据集，实现了四种不同的分类算法：

- **K近邻（KNN）分类器**：比较不同k值对分类性能的影响
- **决策树分类器**：比较剪枝与不剪枝的分类性能
- **逻辑回归分类器**：比较不同正则化强度的性能，分析特征重要性
- **支持向量机（SVM）分类器**：比较不同核函数的性能，调优超参数

### 🏠 房价数据分析

使用自定义房价数据集，实现了两类任务：

#### 📊 房价回归分析
预测房屋总价，实现了多种回归算法：
- 岭回归
- Lasso回归
- 决策树回归
- 随机森林回归
- 梯度提升树回归
- XGBoost回归
- 神经网络回归

比较不同算法的性能指标（MSE、RMSE、MAE、R²）
  
#### 📑 房价分类分析
预测房屋价格类别（低价、中低价、中高价、高价），实现了多种分类算法：
- 逻辑回归
- 决策树
- 随机森林
- 梯度提升决策树
- SVM
- KNN

比较不同算法的性能指标（准确率、精确率、召回率、F1分数）

### 🧹 数据预处理

- ⚠️ 缺失值处理
- 🔎 异常值检测
- 📏 特征标准化
- 🏷️ 类别特征编码
- 📊 自动分段（房价类别划分）

### 📈 可视化分析

项目实现了丰富的可视化功能，所有生成的图表都会自动保存到`plots`目录：

- 混淆矩阵
- ROC曲线
- 回归结果对比
- 回归误差分析
- 特征重要性分析
- 超参数调优结果
- 模型性能比较

## 🛠️ 环境要求

- Python 3.6+
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn
- xgboost（可选，用于高级模型）

## 📥 安装指南

1. 克隆代码库：
```bash
git clone https://github.com/yourusername/ml_project.git
cd ml_project
```

2. 创建并激活conda环境：
```bash
conda create -n ml_project python=3.8
conda activate ml_project
```

3. 安装依赖：
```bash
pip install scikit-learn numpy pandas matplotlib seaborn xgboost
```

## 🚀 使用方法

### 运行所有分析

```bash
python src/main.py
```

### 仅运行乳腺癌分类分析

```bash
python src/main.py --task breast_cancer
```

### 仅运行房价数据分析

```bash
python src/main.py --task real_estate
```

## 📊 输出结果

- **控制台输出**：详细的模型训练和评估结果
- **可视化图表**：所有生成的图表都会保存到`plots`目录
- **最佳模型比较**：程序会自动比较不同算法的性能，并输出最佳模型

## 🔧 扩展项目

### 添加新的数据集

1. 在`src/data/`目录下创建新的数据加载模块
2. 参考现有的数据加载器实现数据加载和预处理功能
3. 在`main.py`中添加对应的分析函数

### 添加新的算法

1. 在`src/models/`目录下的相应子目录中创建新的模型实现
2. 参考现有模型实现训练和评估功能
3. 在`main.py`中集成新的模型

## ❓ 常见问题

详见 [机器学习项目开发中的常见问题与解决方案.md](机器学习项目开发中的常见问题与解决方案.md) 文档，其中包含了在项目开发过程中遇到的典型问题及其解决方案。

## 📄 许可证

[MIT](https://opensource.org/licenses/MIT)
