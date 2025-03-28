# src/models/classification/price_classifiers.py
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils.evaluation import evaluate_classification_model
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    compare_models,
    ensure_plots_dir
)


def train_model(X_train, y_train, model_type='random_forest', **kwargs):
    """
    训练分类模型

    参数:
    X_train: array, 训练特征
    y_train: array, 训练标签
    model_type: str, 模型类型 ('logistic', 'decision_tree', 'random_forest', 'gbdt', 'svm', 'knn')
    **kwargs: 模型参数

    返回:
    model: 训练好的模型
    """
    if model_type == 'logistic':
        model = LogisticRegression(random_state=42, max_iter=1000, **kwargs)
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(random_state=42, **kwargs)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42, **kwargs)
    elif model_type == 'gbdt':
        model = GradientBoostingClassifier(random_state=42, **kwargs)
    elif model_type == 'svm':
        model = SVC(random_state=42, probability=True, **kwargs)
    elif model_type == 'knn':
        model = KNeighborsClassifier(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    model.fit(X_train, y_train)
    return model


def run_price_classification_analysis(X_train, X_test, y_train, y_test, feature_names, category_names):
    """
    运行房价分类分析

    参数:
    X_train, X_test, y_train, y_test: 训练集和测试集
    feature_names: list, 特征名称
    category_names: list, 类别名称

    返回:
    results: dict, 分类分析结果
    """
    print("\n=== 房价分类分析 ===")

    # 准备多个模型
    model_configs = [
        {'type': 'logistic', 'name': '逻辑回归', 'params': {'C': 1.0}},
        {'type': 'decision_tree', 'name': '决策树', 'params': {'max_depth': 5}},
        {'type': 'random_forest', 'name': '随机森林', 'params': {'n_estimators': 100, 'max_depth': 5}},
        {'type': 'gbdt', 'name': '梯度提升决策树', 'params': {'n_estimators': 100, 'learning_rate': 0.1}},
        {'type': 'svm', 'name': 'SVM', 'params': {'kernel': 'rbf', 'C': 1.0}},
        {'type': 'knn', 'name': 'KNN', 'params': {'n_neighbors': 5}}
    ]

    # 存储各模型结果
    model_results = []

    # 训练并评估每个模型
    for config in model_configs:
        print(f"\n训练{config['name']}分类器...")

        # 训练模型
        model = train_model(X_train, y_train, config['type'], **config['params'])

        # 在测试集上预测
        y_pred = model.predict(X_test)

        # 如果模型支持predict_proba，则获取预测概率
        y_score = None
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test)

        # 评估模型
        metrics = evaluate_classification_model(config['name'], y_test, y_pred, y_score, category_names)

        # 绘制混淆矩阵
        plot_confusion_matrix(y_test, y_pred, config['name'], category_names)

        # 如果模型支持特征重要性，则绘制特征重要性图
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(model, feature_names, config['name'])

        # 存储结果
        model_results.append({
            'model_name': config['name'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'model': model
        })

    # 比较各模型的性能
    best_model_name = compare_models(model_results, 'f1', higher_is_better=True)

    # 找出最佳模型
    best_model = next(result['model'] for result in model_results if result['model_name'] == best_model_name)

    return {
        'model_results': model_results,
        'best_model': best_model,
        'best_model_name': best_model_name
    }