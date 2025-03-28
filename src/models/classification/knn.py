# src/models/classification/knn.py
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from src.utils.evaluation import evaluate_classification_model
from src.utils.visualization import plot_confusion_matrix, plot_comparison


def train_knn(X_train, y_train, k=5):
    """
    训练KNN分类器

    参数:
    X_train: array, 训练特征
    y_train: array, 训练标签
    k: int, 近邻数

    返回:
    model: 训练好的KNN模型
    """
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model


def compare_k_values(X_train, X_test, y_train, y_test, k_values=None, class_names=None):
    """
    比较不同k值对KNN性能的影响

    参数:
    X_train, X_test, y_train, y_test: 训练集和测试集
    k_values: list, 要比较的k值列表
    class_names: list, 类别名称

    返回:
    results: dict, 不同k值的性能结果
    best_k: int, 最佳k值
    best_model: 最佳模型
    """
    if k_values is None:
        k_values = [1, 3, 5, 7, 9, 11, 13, 15]

    accuracy_list = []
    f1_list = []
    model_results = []

    for k in k_values:
        print(f"\n评估 k={k}")
        # 训练模型
        model = train_knn(X_train, y_train, k)

        # 在测试集上预测
        y_pred = model.predict(X_test)

        # 如果模型支持predict_proba，则获取预测概率
        y_score = None
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test)
            if len(np.unique(y_test)) == 2:  # 二分类
                y_score = y_score[:, 1]

        # 评估模型
        metrics = evaluate_classification_model(f"KNN (k={k})", y_test, y_pred, y_score, class_names)

        # 绘制混淆矩阵
        plot_confusion_matrix(y_test, y_pred, f"KNN (k={k})", class_names)

        accuracy_list.append(metrics['accuracy'])
        f1_list.append(metrics['f1'])
        metrics['k'] = k
        model_results.append({'k': k, 'metrics': metrics, 'model': model})

    # 绘制不同k值的准确率比较图
    plot_comparison(
        '不同k值的KNN准确率比较',
        'k值',
        '准确率',
        k_values,
        [accuracy_list],
        filename='knn_k_values_comparison_accuracy.png'
    )

    # 绘制不同k值的F1分数比较图
    plot_comparison(
        '不同k值的KNN F1分数比较',
        'k值',
        'F1分数',
        k_values,
        [f1_list],
        filename='knn_k_values_comparison_f1.png'
    )

    # 找出最佳k值（基于F1分数）
    best_idx = np.argmax(f1_list)
    best_k = k_values[best_idx]
    best_model = model_results[best_idx]['model']

    print(f"\n最佳k值为 {best_k}，F1分数: {f1_list[best_idx]:.4f}, 准确率: {accuracy_list[best_idx]:.4f}")

    return {
        'results': model_results,
        'best_k': best_k,
        'best_model': best_model
    }


def run_knn_analysis(X_train, X_test, y_train, y_test, class_names=None):
    """
    运行KNN分析，比较不同k值

    参数:
    X_train, X_test, y_train, y_test: 训练集和测试集
    class_names: list, 类别名称

    返回:
    results: dict, KNN分析结果
    """
    print("\n=== KNN分类器分析 ===")
    return compare_k_values(X_train, X_test, y_train, y_test, class_names=class_names)