# src/main.py
import os
import sys
import argparse
import numpy as np

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.breast_cancer_loader import load_breast_cancer_data
from src.data.real_estate_loader import (
    load_real_estate_data, preprocess_real_estate_data, add_price_category,
    prepare_regression_data, prepare_classification_data
)
from src.models.classification.knn import run_knn_analysis
from src.models.classification.decision_tree import run_decision_tree_analysis
from src.models.classification.linear_model import run_linear_model_analysis
from src.models.classification.svm import run_svm_analysis
from src.models.classification.price_classifiers import run_price_classification_analysis
from src.models.regression.price_regression import run_price_regression_analysis


def run_breast_cancer_classification():
    """运行乳腺癌数据分类任务"""
    print("\n==================== 乳腺癌数据分类任务 ====================")

    # 加载乳腺癌数据
    X_train, X_test, y_train, y_test, feature_names, target_names = load_breast_cancer_data()

    # 问题1: KNN分类
    knn_results = run_knn_analysis(X_train, X_test, y_train, y_test, target_names)

    # 问题2: 决策树分类
    dt_results = run_decision_tree_analysis(X_train, X_test, y_train, y_test, feature_names, target_names)

    # 问题3: 线性模型分类
    lm_results = run_linear_model_analysis(X_train, X_test, y_train, y_test, feature_names, target_names)

    # 问题4: SVM分类
    svm_results = run_svm_analysis(X_train, X_test, y_train, y_test, target_names)

    # 打印结果结构以便调试
    print("\nKNN结果包含的键:", knn_results.keys())
    print("决策树结果包含的键:", dt_results.keys())
    print("线性模型结果包含的键:", lm_results.keys())
    print("SVM结果包含的键:", svm_results.keys())

    # 汇总结果
    print("\n==================== 乳腺癌分类任务汇总结果 ====================")
    print(f"KNN最佳k值: {knn_results['best_k']}")
    print(
        f"决策树是否应该剪枝: {'是' if dt_results['results']['results']['pruned']['f1'] > dt_results['results']['results']['unpruned']['f1'] else '否'}")
    print(f"线性模型最佳正则化参数: C={lm_results['results']['best_C']}")
    print(f"SVM最佳核函数: {svm_results['kernel_results']['best_kernel']}")

    # 现在，让我们根据模型的输出结构来获取F1分数
    # 这里我们需要从每个模型的结果中正确提取F1分数
    # 使用临时变量储存F1分数
    knn_f1 = 0
    dt_f1 = 0
    lm_f1 = 0
    svm_f1 = 0

    # 查找KNN的F1分数
    for result in knn_results['results']:
        if result['k'] == knn_results['best_k']:
            knn_f1 = result['metrics']['f1']
            break

    # 获取决策树的F1分数
    dt_f1 = dt_results['results']['results']['pruned']['f1']

    # 获取线性模型的F1分数 (这里假设最佳模型的性能已经在某处存储)
    for result in lm_results['results']['results']:
        if result['C'] == lm_results['results']['best_C']:
            lm_f1 = result['metrics']['f1']
            break

    # 获取SVM的F1分数
    for result in svm_results['kernel_results']['results']:
        if result['kernel'] == svm_results['kernel_results']['best_kernel']:
            svm_f1 = result['metrics']['f1']
            break

    # 比较各模型F1分数
    models = [
        ('KNN', knn_f1),
        ('决策树', dt_f1),
        ('线性模型', lm_f1),
        ('SVM', svm_f1)
    ]

    # 找出最佳模型
    best_model = max(models, key=lambda x: x[1])
    print(f"\n最佳模型: {best_model[0]}, F1分数: {best_model[1]:.4f}")


def run_real_estate_analysis():
    """运行房价数据分析任务"""
    print("\n==================== 房价数据分析任务 ====================")

    # 加载房价数据
    df = load_real_estate_data()

    # 数据预处理
    df, encoders = preprocess_real_estate_data(df)

    # 添加价格类别
    df, category_names = add_price_category(df)

    # 问题5: 房价回归
    X_train_reg, X_test_reg, y_train_reg, y_test_reg, feature_names_reg, scaler_reg = prepare_regression_data(df)
    reg_results = run_price_regression_analysis(X_train_reg, X_test_reg, y_train_reg, y_test_reg, feature_names_reg)

    # 问题6: 房价分类
    X_train_cls, X_test_cls, y_train_cls, y_test_cls, feature_names_cls, scaler_cls = prepare_classification_data(df)
    cls_results = run_price_classification_analysis(X_train_cls, X_test_cls, y_train_cls, y_test_cls, feature_names_cls,
                                                    category_names)

    # 汇总结果
    print("\n==================== 房价分析任务汇总结果 ====================")
    print(f"最佳回归模型 (基于R²): {reg_results['best_model_name_r2']}")
    print(f"最佳回归模型 (基于RMSE): {reg_results['best_model_name_rmse']}")
    print(f"最佳分类模型: {cls_results['best_model_name']}")


def main():
    parser = argparse.ArgumentParser(description="机器学习模型分析工具")
    parser.add_argument('--task', type=str, choices=['breast_cancer', 'real_estate', 'all'],
                        default='all', help='要运行的任务，可选：breast_cancer, real_estate, all')

    args = parser.parse_args()

    if args.task == 'breast_cancer' or args.task == 'all':
        run_breast_cancer_classification()

    if args.task == 'real_estate' or args.task == 'all':
        run_real_estate_analysis()

    print("\n所有任务完成！")


if __name__ == "__main__":
    main()