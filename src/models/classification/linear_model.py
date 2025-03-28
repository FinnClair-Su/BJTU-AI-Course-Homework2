# src/models/classification/linear_model.py
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import os
from src.utils.evaluation import evaluate_classification_model
from src.utils.visualization import plot_confusion_matrix, ensure_plots_dir


def train_logistic_regression(X_train, y_train, C=1.0):
    """
    训练逻辑回归分类器

    参数:
    X_train: array, 训练特征
    y_train: array, 训练标签
    C: float, 正则化强度的倒数

    返回:
    model: 训练好的逻辑回归模型
    """
    model = LogisticRegression(C=C, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def compare_regularization(X_train, X_test, y_train, y_test, class_names=None):
    """
    比较不同正则化强度的逻辑回归

    参数:
    X_train, X_test, y_train, y_test: 训练集和测试集
    class_names: list, 类别名称

    返回:
    results: dict, 不同正则化强度的性能结果
    best_model: 最佳模型
    """
    # 测试不同的正则化强度
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    accuracy_list = []
    f1_list = []
    model_results = []

    for C in C_values:
        print(f"\n评估 C={C}")
        # 训练模型
        model = train_logistic_regression(X_train, y_train, C)

        # 在测试集上预测
        y_pred = model.predict(X_test)

        # 如果模型支持predict_proba，则获取预测概率
        y_score = None
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test)
            if len(np.unique(y_test)) == 2:  # 二分类
                y_score = y_score[:, 1]

        # 评估模型
        metrics = evaluate_classification_model(f"逻辑回归 (C={C})", y_test, y_pred, y_score, class_names)
        plot_confusion_matrix(y_test, y_pred, f"逻辑回归 (C={C})", class_names)

        accuracy_list.append(metrics['accuracy'])
        f1_list.append(metrics['f1'])
        model_results.append({'C': C, 'metrics': metrics, 'model': model})

    # 绘制不同C值的准确率比较图
    plt.figure(figsize=(10, 6))
    plt.semilogx(C_values, accuracy_list, marker='o', label='准确率')
    plt.semilogx(C_values, f1_list, marker='s', label='F1分数')
    plt.xlabel('正则化参数 C (对数刻度)')
    plt.ylabel('性能')
    plt.title('不同正则化强度的逻辑回归性能')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 保存图表
    plots_dir = ensure_plots_dir()
    save_path = os.path.join(plots_dir, 'logistic_regression_C_values.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"逻辑回归正则化参数图已保存至: {save_path}")

    plt.show()

    # 找出最佳C值（基于F1分数）
    best_idx = np.argmax(f1_list)
    best_C = C_values[best_idx]
    best_model = model_results[best_idx]['model']

    print(f"\n最佳正则化参数 C={best_C}，F1分数: {f1_list[best_idx]:.4f}, 准确率: {accuracy_list[best_idx]:.4f}")

    return {
        'results': model_results,
        'best_C': best_C,
        'best_model': best_model
    }


def analyze_feature_importance(model, feature_names):
    """
    分析特征重要性

    参数:
    model: 逻辑回归模型
    feature_names: list, 特征名称
    """
    # 获取特征系数
    if len(model.classes_) == 2:  # 二分类
        coefficients = model.coef_[0]
    else:  # 多分类
        # 平均每个类别的系数绝对值
        coefficients = np.mean(np.abs(model.coef_), axis=0)

    # 结合特征名称和系数
    feature_importance = list(zip(feature_names, coefficients))

    # 按绝对值排序
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

    # 提取前10个最重要的特征
    top_features = feature_importance[:10]

    # 绘制特征重要性
    plt.figure(figsize=(12, 8))
    features, importance = zip(*top_features)
    colors = ['red' if i < 0 else 'blue' for i in importance]
    plt.barh(range(len(features)), [abs(i) for i in importance], color=colors)
    plt.yticks(range(len(features)), features)
    plt.xlabel('系数绝对值')
    plt.title('逻辑回归模型的前10个重要特征')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='正相关'),
        Patch(facecolor='red', label='负相关')
    ]
    plt.legend(handles=legend_elements)

    plt.tight_layout()

    # 保存图表
    plots_dir = ensure_plots_dir()
    save_path = os.path.join(plots_dir, 'logistic_regression_feature_importance.png')
    plt.savefig(save_path, dpi=300)
    print(f"特征重要性图已保存至: {save_path}")

    plt.show()


def run_linear_model_analysis(X_train, X_test, y_train, y_test, feature_names, class_names=None):
    """
    运行线性模型分析

    参数:
    X_train, X_test, y_train, y_test: 训练集和测试集
    feature_names: list, 特征名称
    class_names: list, 类别名称

    返回:
    results: dict, 线性模型分析结果
    """
    print("\n=== 线性模型分析 ===")

    # 比较不同正则化强度
    model_results = compare_regularization(X_train, X_test, y_train, y_test, class_names)

    # 找出最佳C值
    best_C = model_results['best_C']
    best_model = model_results['best_model']

    # 分析特征重要性
    analyze_feature_importance(best_model, feature_names)

    return {
        'results': model_results,
        'best_model': best_model
    }