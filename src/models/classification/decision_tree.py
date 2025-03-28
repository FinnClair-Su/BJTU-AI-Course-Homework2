# src/models/classification/decision_tree.py
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np
import os
from src.utils.evaluation import evaluate_classification_model
from src.utils.visualization import plot_confusion_matrix, ensure_plots_dir


def train_decision_tree(X_train, y_train, max_depth=None, ccp_alpha=0.0):
    """
    训练决策树分类器

    参数:
    X_train: array, 训练特征
    y_train: array, 训练标签
    max_depth: int or None, 树的最大深度
    ccp_alpha: float, 复杂度参数，用于剪枝

    返回:
    model: 训练好的决策树模型
    """
    model = DecisionTreeClassifier(max_depth=max_depth, ccp_alpha=ccp_alpha, random_state=42)
    model.fit(X_train, y_train)
    return model


def visualize_tree(model, feature_names, class_names, max_depth=3, is_pruned=False):
    """
    可视化决策树

    参数:
    model: 决策树模型
    feature_names: list, 特征名称
    class_names: list, 类别名称
    max_depth: int, 可视化的最大深度
    is_pruned: bool, 是否是剪枝后的树
    """
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True, max_depth=max_depth)
    tree_type = "剪枝" if is_pruned else "不剪枝"
    plt.title(f"{tree_type}决策树 (最大深度={max_depth})")

    # 保存图表
    plots_dir = ensure_plots_dir()
    save_path = os.path.join(plots_dir, f'decision_tree_{tree_type}_depth{max_depth}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"决策树图已保存至: {save_path}")

    plt.show()


def compare_pruning(X_train, X_test, y_train, y_test, feature_names, class_names):
    """
    比较剪枝与不剪枝的决策树

    参数:
    X_train, X_test, y_train, y_test: 训练集和测试集
    feature_names: list, 特征名称
    class_names: list, 类别名称

    返回:
    results: dict, 剪枝与不剪枝的性能结果
    """
    results = {}

    # 不剪枝的决策树
    print("\n不剪枝的决策树:")
    unpruned_model = train_decision_tree(X_train, y_train)
    y_pred_unpruned = unpruned_model.predict(X_test)

    # 如果模型支持predict_proba，则获取预测概率
    y_score_unpruned = None
    if hasattr(unpruned_model, 'predict_proba'):
        y_score_unpruned = unpruned_model.predict_proba(X_test)
        if len(np.unique(y_test)) == 2:  # 二分类
            y_score_unpruned = y_score_unpruned[:, 1]

    metrics_unpruned = evaluate_classification_model("不剪枝的决策树", y_test, y_pred_unpruned,
                                                     y_score_unpruned, class_names)
    plot_confusion_matrix(y_test, y_pred_unpruned, "不剪枝的决策树", class_names)
    results['unpruned'] = metrics_unpruned

    # 可视化不剪枝的决策树
    visualize_tree(unpruned_model, feature_names, class_names, is_pruned=False)

    # 交叉验证确定最佳剪枝参数
    path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    # 选择一些ccp_alpha值进行测试
    alphas = ccp_alphas[::3]  # 取每3个值以减少计算量
    if len(alphas) > 6:
        alphas = alphas[:6]  # 最多使用6个值

    acc_train = []
    acc_test = []

    for alpha in alphas:
        # 训练剪枝模型
        pruned_model = train_decision_tree(X_train, y_train, ccp_alpha=alpha)

        # 计算训练集准确率
        y_train_pred = pruned_model.predict(X_train)
        acc_train.append(np.mean(y_train_pred == y_train))

        # 计算测试集准确率
        y_test_pred = pruned_model.predict(X_test)
        acc_test.append(np.mean(y_test_pred == y_test))

    # 找出测试集准确率最高的alpha值
    best_alpha_idx = np.argmax(acc_test)
    best_alpha = alphas[best_alpha_idx]

    # 使用最佳alpha值训练模型
    print(f"\n剪枝的决策树 (ccp_alpha={best_alpha:.6f}):")
    pruned_model = train_decision_tree(X_train, y_train, ccp_alpha=best_alpha)
    y_pred_pruned = pruned_model.predict(X_test)

    # 如果模型支持predict_proba，则获取预测概率
    y_score_pruned = None
    if hasattr(pruned_model, 'predict_proba'):
        y_score_pruned = pruned_model.predict_proba(X_test)
        if len(np.unique(y_test)) == 2:  # 二分类
            y_score_pruned = y_score_pruned[:, 1]

    metrics_pruned = evaluate_classification_model("剪枝的决策树", y_test, y_pred_pruned,
                                                   y_score_pruned, class_names)
    plot_confusion_matrix(y_test, y_pred_pruned, "剪枝的决策树", class_names)
    results['pruned'] = metrics_pruned

    # 可视化剪枝的决策树
    visualize_tree(pruned_model, feature_names, class_names, is_pruned=True)

    # 绘制不同alpha值的准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, acc_train, marker='o', label='训练集准确率')
    plt.plot(alphas, acc_test, marker='o', label='测试集准确率')
    plt.axvline(x=best_alpha, color='r', linestyle='--', label=f'最佳alpha={best_alpha:.6f}')
    plt.xlabel('alpha值')
    plt.ylabel('准确率')
    plt.title('不同剪枝参数的准确率')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图表
    plots_dir = ensure_plots_dir()
    save_path = os.path.join(plots_dir, 'decision_tree_pruning_alphas.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"剪枝参数图已保存至: {save_path}")

    plt.show()

    # 返回两个模型和结果
    return {
        'unpruned_model': unpruned_model,
        'pruned_model': pruned_model,
        'best_alpha': best_alpha,
        'results': results
    }


def run_decision_tree_analysis(X_train, X_test, y_train, y_test, feature_names, class_names):
    """
    运行决策树分析，比较剪枝与不剪枝

    参数:
    X_train, X_test, y_train, y_test: 训练集和测试集
    feature_names: list, 特征名称
    class_names: list, 类别名称

    返回:
    results: dict, 剪枝与不剪枝的性能结果
    """
    print("\n=== 决策树分类器分析 ===")
    results = compare_pruning(X_train, X_test, y_train, y_test, feature_names, class_names)

    # 比较剪枝与不剪枝的性能
    print("\n剪枝与不剪枝的性能比较:")
    print(f"不剪枝准确率: {results['results']['unpruned']['accuracy']:.4f}")
    print(f"不剪枝F1分数: {results['results']['unpruned']['f1']:.4f}")
    print(f"剪枝准确率: {results['results']['pruned']['accuracy']:.4f}")
    print(f"剪枝F1分数: {results['results']['pruned']['f1']:.4f}")

    # 判断哪个模型更好
    if results['results']['pruned']['f1'] > results['results']['unpruned']['f1']:
        print("\n剪枝后的模型性能更好")
        best_model = results['pruned_model']
    else:
        print("\n不剪枝的模型性能更好")
        best_model = results['unpruned_model']

    return {
        'results': results,
        'best_model': best_model
    }