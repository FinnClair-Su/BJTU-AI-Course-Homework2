# src/utils/evaluation.py
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)


def evaluate_regression_model(model_name, y_true, y_pred):
    """
    评估回归模型性能

    参数:
    model_name: str, 模型名称
    y_true: array, 真实值
    y_pred: array, 预测值

    返回:
    metrics: dict, 包含MSE, MAE, R2等指标
    """
    # 计算评估指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # 打印评估结果
    print(f"\n{model_name} 评估结果:")
    print(f"  均方误差 (MSE): {mse:.4f}")
    print(f"  均方根误差 (RMSE): {rmse:.4f}")
    print(f"  平均绝对误差 (MAE): {mae:.4f}")
    print(f"  决定系数 (R²): {r2:.4f}")

    return {
        'model_name': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def evaluate_classification_model(model_name, y_true, y_pred, y_score=None, class_names=None):
    """
    评估分类模型性能

    参数:
    model_name: str, 模型名称
    y_true: array, 真实标签
    y_pred: array, 预测标签
    y_score: array, 各类别的预测概率 (用于ROC曲线)
    class_names: list, 类别名称

    返回:
    metrics: dict, 包含准确率, 精确率, 召回率, F1等指标
    """
    # 计算评估指标
    accuracy = accuracy_score(y_true, y_pred)

    # 多分类情况处理
    if len(np.unique(y_true)) > 2:
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
    else:
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

    # 打印评估结果
    print(f"\n{model_name} 评估结果:")
    print(f"  准确率 (Accuracy): {accuracy:.4f}")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall): {recall:.4f}")
    print(f"  F1分数: {f1:.4f}")

    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 计算ROC曲线和AUC (二分类情况)
    roc_auc = None
    if y_score is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        print(f"  ROC AUC: {roc_auc:.4f}")

    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'roc_auc': roc_auc
    }