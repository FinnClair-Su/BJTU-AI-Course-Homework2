# src/models/classification/svm.py
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils.evaluation import evaluate_classification_model
from src.utils.visualization import plot_confusion_matrix, ensure_plots_dir


def train_svm(X_train, y_train, kernel='linear', C=1.0, gamma='scale'):
    """
    训练SVM分类器

    参数:
    X_train: array, 训练特征
    y_train: array, 训练标签
    kernel: str, 核函数类型
    C: float, 正则化参数
    gamma: str or float, 'rbf', 'poly' 和 'sigmoid' 核函数的参数

    返回:
    model: 训练好的SVM模型
    """
    model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42, probability=True)
    model.fit(X_train, y_train)
    return model


def compare_kernels(X_train, X_test, y_train, y_test, class_names=None):
    """
    比较不同核函数的SVM性能

    参数:
    X_train, X_test, y_train, y_test: 训练集和测试集
    class_names: list, 类别名称

    返回:
    results: dict, 不同核函数的性能结果
    best_kernel: str, 最佳核函数
    best_model: 最佳模型
    """
    # 测试不同的核函数
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    accuracy_list = []
    f1_list = []
    model_results = []

    for kernel in kernels:
        print(f"\n评估核函数: {kernel}")
        # 训练模型
        model = train_svm(X_train, y_train, kernel=kernel)

        # 在测试集上预测
        y_pred = model.predict(X_test)

        # 如果模型支持predict_proba，则获取预测概率
        y_score = None
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test)
            if len(np.unique(y_test)) == 2:  # 二分类
                y_score = y_score[:, 1]

        # 评估模型
        metrics = evaluate_classification_model(f"SVM ({kernel}核)", y_test, y_pred, y_score, class_names)
        plot_confusion_matrix(y_test, y_pred, f"SVM ({kernel}核)", class_names)

        accuracy_list.append(metrics['accuracy'])
        f1_list.append(metrics['f1'])
        model_results.append({'kernel': kernel, 'metrics': metrics, 'model': model})

    # 绘制不同核函数的准确率比较图
    plt.figure(figsize=(10, 6))
    plt.bar(kernels, accuracy_list, color='skyblue', alpha=0.7, label='准确率')
    plt.bar(kernels, f1_list, color='salmon', alpha=0.7, label='F1分数')
    plt.xlabel('核函数')
    plt.ylabel('性能')
    plt.title('不同核函数的SVM性能')
    plt.ylim(0.5, 1.0)  # 调整y轴范围以便更好地显示差异
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.legend()

    # 添加准确率标签
    for i, acc in enumerate(accuracy_list):
        plt.text(i, acc + 0.02, f'{acc:.4f}', ha='center')

    # 添加F1分数标签
    for i, f1 in enumerate(f1_list):
        plt.text(i, f1 - 0.04, f'{f1:.4f}', ha='center')

    plt.tight_layout()

    # 保存图表
    plots_dir = ensure_plots_dir()
    save_path = os.path.join(plots_dir, 'svm_kernel_comparison.png')
    plt.savefig(save_path, dpi=300)
    print(f"SVM核函数比较图已保存至: {save_path}")

    plt.show()

    # 找出最佳核函数（基于F1分数）
    best_idx = np.argmax(f1_list)
    best_kernel = kernels[best_idx]
    best_model = model_results[best_idx]['model']

    print(f"\n最佳核函数: {best_kernel}，F1分数: {f1_list[best_idx]:.4f}, 准确率: {accuracy_list[best_idx]:.4f}")

    return {
        'results': model_results,
        'best_kernel': best_kernel,
        'best_model': best_model
    }


def tune_hyperparameters(X_train, X_test, y_train, y_test, kernel='rbf', class_names=None):
    """
    调优SVM超参数

    参数:
    X_train, X_test, y_train, y_test: 训练集和测试集
    kernel: str, 要调优的核函数
    class_names: list, 类别名称

    返回:
    best_params: dict, 最佳参数组合
    best_model: 最佳模型
    """
    # 测试不同的C值和gamma值
    C_values = [0.1, 1, 10, 100]
    gamma_values = ['scale', 'auto', 0.01, 0.1, 1]

    best_accuracy = 0
    best_f1 = 0
    best_C = None
    best_gamma = None
    best_model = None

    results = np.zeros((len(C_values), len(gamma_values)))
    f1_scores = np.zeros((len(C_values), len(gamma_values)))

    for i, C in enumerate(C_values):
        for j, gamma in enumerate(gamma_values):
            # 训练模型
            model = train_svm(X_train, y_train, kernel=kernel, C=C, gamma=gamma)

            # 在测试集上预测
            y_pred = model.predict(X_test)

            # 计算准确率和F1分数
            metrics = evaluate_classification_model(
                f"SVM ({kernel}, C={C}, gamma={gamma})",
                y_test, y_pred, None, class_names
            )

            accuracy = metrics['accuracy']
            f1 = metrics['f1']

            results[i, j] = accuracy
            f1_scores[i, j] = f1

            # 更新最佳参数（基于F1分数）
            if f1 > best_f1:
                best_f1 = f1
                best_accuracy = accuracy
                best_C = C
                best_gamma = gamma
                best_model = model

    # 绘制热力图
    plt.figure(figsize=(12, 10))

    # 准确率热力图
    plt.subplot(2, 1, 1)
    plt.imshow(results, interpolation='nearest', cmap='viridis')
    plt.title(f'{kernel}核SVM的超参数调优 (准确率)')
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar(label='准确率')
    plt.xticks(np.arange(len(gamma_values)), gamma_values)
    plt.yticks(np.arange(len(C_values)), C_values)

    # 标记最佳参数
    best_i_acc = np.unravel_index(np.argmax(results), results.shape)[0]
    best_j_acc = np.unravel_index(np.argmax(results), results.shape)[1]
    plt.plot(best_j_acc, best_i_acc, 'r*', markersize=15)

    # 添加文本注释
    for i in range(len(C_values)):
        for j in range(len(gamma_values)):
            plt.text(j, i, f'{results[i, j]:.4f}',
                     ha="center", va="center", color="w")

    # F1分数热力图
    plt.subplot(2, 1, 2)
    plt.imshow(f1_scores, interpolation='nearest', cmap='viridis')
    plt.title(f'{kernel}核SVM的超参数调优 (F1分数)')
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar(label='F1分数')
    plt.xticks(np.arange(len(gamma_values)), gamma_values)
    plt.yticks(np.arange(len(C_values)), C_values)

    # 标记最佳参数
    best_i_f1 = np.unravel_index(np.argmax(f1_scores), f1_scores.shape)[0]
    best_j_f1 = np.unravel_index(np.argmax(f1_scores), f1_scores.shape)[1]
    plt.plot(best_j_f1, best_i_f1, 'r*', markersize=15)

    # 添加文本注释
    for i in range(len(C_values)):
        for j in range(len(gamma_values)):
            plt.text(j, i, f'{f1_scores[i, j]:.4f}',
                     ha="center", va="center", color="w")

    plt.tight_layout()

    # 保存图表
    plots_dir = ensure_plots_dir()
    save_path = os.path.join(plots_dir, f'svm_{kernel}_hyperparameter_tuning.png')
    plt.savefig(save_path, dpi=300)
    print(f"SVM {kernel}核超参数调优图已保存至: {save_path}")

    plt.show()

    print(f"\n最佳参数: C={best_C}, gamma={best_gamma}")
    print(f"最佳F1分数: {best_f1:.4f}")
    print(f"对应准确率: {best_accuracy:.4f}")

    return {
        'C': best_C,
        'gamma': best_gamma,
        'f1': best_f1,
        'accuracy': best_accuracy,
        'model': best_model
    }


def run_svm_analysis(X_train, X_test, y_train, y_test, class_names=None):
    """
    运行SVM分析，比较不同核函数

    参数:
    X_train, X_test, y_train, y_test: 训练集和测试集
    class_names: list, 类别名称

    返回:
    results: dict, SVM分析结果
    """
    print("\n=== SVM分类器分析 ===")

    # 比较不同核函数
    kernel_results = compare_kernels(X_train, X_test, y_train, y_test, class_names)

    # 找出最佳核函数
    best_kernel = kernel_results['best_kernel']
    print(f"\n最佳核函数: {best_kernel}")

    # 对线性核和最佳非线性核进行超参数调优
    print("\n对线性核进行超参数调优:")
    linear_best_params = tune_hyperparameters(X_train, X_test, y_train, y_test, kernel='linear',
                                              class_names=class_names)

    if best_kernel != 'linear':
        print(f"\n对{best_kernel}核函数进行超参数调优:")
        nonlinear_best_params = tune_hyperparameters(X_train, X_test, y_train, y_test, kernel=best_kernel,
                                                     class_names=class_names)

        # 比较最佳线性核和非线性核的性能
        print("\n线性核与非线性核的最佳性能比较:")
        print(f"最佳线性核F1分数: {linear_best_params['f1']:.4f}")
        print(f"最佳{best_kernel}核F1分数: {nonlinear_best_params['f1']:.4f}")

        # 选择性能更好的模型
        if nonlinear_best_params['f1'] > linear_best_params['f1']:
            print(f"非线性核({best_kernel})性能更好")
            best_model = nonlinear_best_params['model']
        else:
            print("线性核性能更好")
            best_model = linear_best_params['model']
    else:
        best_model = linear_best_params['model']

    return {
        'kernel_results': kernel_results,
        'linear_best_params': linear_best_params,
        'best_model': best_model
    }