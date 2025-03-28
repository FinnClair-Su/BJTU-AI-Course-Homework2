# src/data/breast_cancer_loader.py
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_breast_cancer_data(test_size=0.2, random_state=42):
    """
    加载乳腺癌数据集并进行预处理

    参数:
    test_size: float, 测试集比例
    random_state: int, 随机种子

    返回:
    X_train, X_test, y_train, y_test: 训练集和测试集的特征和标签
    feature_names: 特征名称
    target_names: 目标类别名称
    """
    # 加载数据集
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"乳腺癌数据集加载完成:")
    print(f"  训练集样本数: {X_train.shape[0]}")
    print(f"  测试集样本数: {X_test.shape[0]}")
    print(f"  特征数: {X_train.shape[1]}")
    print(f"  类别: {target_names}")

    return X_train, X_test, y_train, y_test, feature_names, target_names