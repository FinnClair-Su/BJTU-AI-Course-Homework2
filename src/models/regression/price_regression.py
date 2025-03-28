# src/models/regression/price_regression.py
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils.evaluation import evaluate_regression_model
from src.utils.visualization import (
    plot_regression_results,
    plot_regression_errors,
    plot_feature_importance,
    compare_models,
    ensure_plots_dir
)


def train_model(X_train, y_train, model_type='random_forest', **kwargs):
    """
    训练回归模型

    参数:
    X_train: array, 训练特征
    y_train: array, 训练标签
    model_type: str, 模型类型 ('ridge', 'lasso', 'decision_tree', 'random_forest', 'gbdt', 'svr', 'mlp', 'xgboost')
    **kwargs: 模型参数

    返回:
    model: 训练好的模型
    """
    if model_type == 'ridge':
        model = Ridge(random_state=42, **kwargs)
    elif model_type == 'lasso':
        model = Lasso(random_state=42, **kwargs)
    elif model_type == 'decision_tree':
        model = DecisionTreeRegressor(random_state=42, **kwargs)
    elif model_type == 'random_forest':
        model = RandomForestRegressor(random_state=42, **kwargs)
    elif model_type == 'gbdt':
        model = GradientBoostingRegressor(random_state=42, **kwargs)
    elif model_type == 'svr':
        model = SVR(**kwargs)
    elif model_type == 'mlp':
        model = MLPRegressor(random_state=42, max_iter=1000, **kwargs)
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(random_state=42, **kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    model.fit(X_train, y_train)
    return model


def run_price_regression_analysis(X_train, X_test, y_train, y_test, feature_names):
    """
    运行房价回归分析

    参数:
    X_train, X_test, y_train, y_test: 训练集和测试集
    feature_names: list, 特征名称

    返回:
    results: dict, 回归分析结果
    """
    print("\n=== 房价回归分析 ===")

    # 准备多个模型
    model_configs = [
        {'type': 'ridge', 'name': '岭回归', 'params': {'alpha': 1.0}},
        {'type': 'lasso', 'name': 'Lasso回归', 'params': {'alpha': 0.1}},
        {'type': 'decision_tree', 'name': '决策树回归', 'params': {'max_depth': 10}},
        {'type': 'random_forest', 'name': '随机森林回归', 'params': {'n_estimators': 100, 'max_depth': 10}},
        {'type': 'gbdt', 'name': '梯度提升决策树回归', 'params': {'n_estimators': 100, 'learning_rate': 0.1}},
        {'type': 'xgboost', 'name': 'XGBoost回归',
         'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5}},
        {'type': 'mlp', 'name': '神经网络回归', 'params': {'hidden_layer_sizes': (100, 50), 'early_stopping': True}}
    ]

    # 存储各模型结果
    model_results = []

    # 训练并评估每个模型
    for config in model_configs:
        print(f"\n训练{config['name']}模型...")

        # 训练模型
        model = train_model(X_train, y_train, config['type'], **config['params'])

        # 在测试集上预测
        y_pred = model.predict(X_test)

        # 评估模型
        metrics = evaluate_regression_model(config['name'], y_test, y_pred)

        # 绘制回归结果图
        plot_regression_results(y_test, y_pred, config['name'])

        # 绘制回归误差图
        plot_regression_errors(y_test, y_pred, config['name'])

        # 如果模型支持特征重要性，则绘制特征重要性图
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(model, feature_names, config['name'])

        # 存储结果
        model_results.append({
            'model_name': config['name'],
            'mse': metrics['mse'],
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2'],
            'model': model
        })

    # 比较各模型的性能
    best_model_name_r2 = compare_models(model_results, 'r2', higher_is_better=True)
    best_model_name_rmse = compare_models(model_results, 'rmse', higher_is_better=False)

    # 用R2作为最终指标
    best_model = next(result['model'] for result in model_results if result['model_name'] == best_model_name_r2)

    return {
        'model_results': model_results,
        'best_model': best_model,
        'best_model_name_r2': best_model_name_r2,
        'best_model_name_rmse': best_model_name_rmse
    }