# src/utils/visualization.py
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

# 设置matplotlib中文支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'
except:
    pass


def ensure_plots_dir():
    """
    确保plots目录存在

    返回:
    plots_dir: str, plots目录路径
    """
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    plots_dir = os.path.join(current_dir, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    return plots_dir


def plot_confusion_matrix(y_true, y_pred, model_name, class_names=None):
    """
    绘制混淆矩阵

    参数:
    y_true: array, 真实标签
    y_pred: array, 预测标签
    model_name: str, 模型名称
    class_names: list, 类别名称
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    if class_names is not None:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.title(f'{model_name} 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')

    # 保存图表
    plots_dir = ensure_plots_dir()
    model_name_safe = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    save_path = os.path.join(plots_dir, f'{model_name_safe}_confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"混淆矩阵已保存至: {save_path}")

    plt.show()


def plot_roc_curve(y_true, y_score, model_name):
    """
    绘制ROC曲线

    参数:
    y_true: array, 真实标签
    y_score: array, 预测概率
    model_name: str, 模型名称
    """
    # 仅适用于二分类
    if len(np.unique(y_true)) != 2:
        print("ROC曲线仅适用于二分类问题")
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (FPR)')
    plt.ylabel('真正例率 (TPR)')
    plt.title(f'{model_name} ROC曲线')
    plt.legend(loc="lower right")

    # 保存图表
    plots_dir = ensure_plots_dir()
    model_name_safe = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    save_path = os.path.join(plots_dir, f'{model_name_safe}_roc_curve.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"ROC曲线已保存至: {save_path}")

    plt.show()


def plot_regression_results(y_true, y_pred, model_name):
    """
    绘制回归模型结果

    参数:
    y_true: array, 真实值
    y_pred: array, 预测值
    model_name: str, 模型名称
    """
    from sklearn.metrics import mean_squared_error, r2_score

    plt.figure(figsize=(10, 8))

    # 预测值 vs 真实值散点图
    plt.scatter(y_true, y_pred, alpha=0.5)

    # 添加对角线 (理想预测)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.title(f'{model_name} 预测值 vs 真实值')
    plt.xlabel('真实值 (万元)')
    plt.ylabel('预测值 (万元)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加文本注释（评估指标）
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    plt.text(
        0.05, 0.95,
        f'RMSE: {rmse:.2f}\nR²: {r2:.4f}',
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    )

    # 保存图表
    plots_dir = ensure_plots_dir()
    model_name_safe = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    save_path = os.path.join(plots_dir, f'regression_{model_name_safe}_results.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"回归结果图已保存至: {save_path}")

    plt.show()


def plot_regression_errors(y_true, y_pred, model_name):
    """
    绘制回归误差分析图

    参数:
    y_true: array, 真实值
    y_pred: array, 预测值
    model_name: str, 模型名称
    """
    errors = y_pred - y_true

    plt.figure(figsize=(10, 8))

    # 误差分布直方图
    plt.hist(errors, bins=30, alpha=0.7, color='skyblue')
    plt.axvline(x=0, color='r', linestyle='--')

    plt.title(f'{model_name} 预测误差分布')
    plt.xlabel('预测误差 (万元)')
    plt.ylabel('频数')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加统计信息
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    plt.text(
        0.05, 0.95,
        f'均值: {mean_error:.2f}\n标准差: {std_error:.2f}',
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    )

    # 保存图表
    plots_dir = ensure_plots_dir()
    model_name_safe = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    save_path = os.path.join(plots_dir, f'regression_{model_name_safe}_errors.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"回归误差图已保存至: {save_path}")

    plt.show()


def plot_feature_importance(model, feature_names, model_name, top_n=10):
    """
    绘制特征重要性图

    参数:
    model: 模型对象，必须有feature_importances_属性
    feature_names: list, 特征名称
    model_name: str, 模型名称
    top_n: int, 显示前n个重要特征
    """
    # 检查模型是否有feature_importances_属性
    if not hasattr(model, 'feature_importances_'):
        print(f"{model_name}没有feature_importances_属性，无法绘制特征重要性图")
        return

    # 获取特征重要性
    importances = model.feature_importances_

    # 对特征重要性进行排序
    indices = np.argsort(importances)[::-1]

    # 选择前top_n个特征
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_features = [feature_names[i] for i in top_indices]

    # 绘制特征重要性图
    plt.figure(figsize=(10, 8))
    plt.title(f'{model_name} 特征重要性')
    plt.barh(range(len(top_importances)), top_importances, color='skyblue')
    plt.yticks(range(len(top_importances)), top_features)
    plt.xlabel('重要性')
    plt.gca().invert_yaxis()  # 最重要的特征显示在顶部

    # 保存图表
    plots_dir = ensure_plots_dir()
    model_name_safe = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    save_path = os.path.join(plots_dir, f'{model_name_safe}_feature_importance.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"特征重要性图已保存至: {save_path}")

    plt.show()


def plot_comparison(title, x_label, y_label, x_values, y_values_list, labels=None, filename=None):
    """
    绘制比较图

    参数:
    title: str, 图表标题
    x_label: str, x轴标签
    y_label: str, y轴标签
    x_values: list, x轴值
    y_values_list: list of lists, 多组y轴值
    labels: list, 图例标签
    filename: str, 保存的文件名（不包含路径）
    """
    plt.figure(figsize=(10, 6))

    if labels:
        for i, y_vals in enumerate(y_values_list):
            plt.plot(x_values, y_vals, marker='o', label=labels[i])
        plt.legend()
    else:
        plt.plot(x_values, y_values_list[0], marker='o')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图表
    plots_dir = ensure_plots_dir()
    if filename is None:
        # 如果未提供文件名，使用标题生成
        filename = title.replace(' ', '_').lower() + '.png'
    save_path = os.path.join(plots_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"图表已保存至: {save_path}")

    plt.show()


def compare_models(model_results, metric_name, higher_is_better=True):
    """
    比较多个模型的性能

    参数:
    model_results: list of dict, 多个模型的评估结果
    metric_name: str, 要比较的指标名称
    higher_is_better: bool, 指标值越高越好
    """
    if not model_results:
        print("没有模型结果可供比较")
        return

    # 提取模型名称和指标值
    model_names = [result['model_name'] for result in model_results]
    metric_values = [result[metric_name] for result in model_results]

    # 绘制比较图
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, metric_values, color='skyblue')
    plt.title(f'模型{metric_name}比较')
    plt.xlabel('模型')
    plt.ylabel(metric_name)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')

    # 在每个柱子上添加数值标签
    for i, v in enumerate(metric_values):
        plt.text(i, v + (v * 0.02 if higher_is_better else -v * 0.08), f'{v:.4f}', ha='center')

    # 找出最佳模型
    if higher_is_better:
        best_model_idx = np.argmax(metric_values)
    else:
        best_model_idx = np.argmin(metric_values)

    best_model = model_names[best_model_idx]
    plt.title(f'模型{metric_name}比较 (最佳模型: {best_model})')

    # 保存图表
    plots_dir = ensure_plots_dir()
    save_path = os.path.join(plots_dir, f'model_comparison_{metric_name}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"模型比较图已保存至: {save_path}")

    plt.show()

    return best_model