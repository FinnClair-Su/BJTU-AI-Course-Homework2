# src/data/real_estate_loader.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


def load_real_estate_data(file_path="data.xlsx"):
    """
    加载房价数据

    参数:
    file_path: str, Excel文件路径

    返回:
    df: DataFrame, 预处理后的数据
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        # 尝试相对路径
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        file_path = os.path.join(base_dir, "data.xlsx")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"无法找到数据文件: {file_path}")

    # 加载数据
    df = pd.read_excel(file_path)

    # 打印原始数据信息
    print("房价数据集加载完成:")
    print(f"  数据形状: {df.shape}")
    print(f"  列名: {df.columns.tolist()}")

    # 删除重复的省市区列
    if '省' in df.columns and '市' in df.columns and '区（县）' in df.columns:
        df = df.drop(['省', '市', '区（县）'], axis=1)

    return df


def preprocess_real_estate_data(df):
    """
    房价数据预处理

    参数:
    df: DataFrame, 原始数据

    返回:
    df: DataFrame, 预处理后的数据
    encoders: dict, 类别特征编码器
    """
    # 处理缺失值
    print(f"\n缺失值统计:\n{df.isnull().sum()}")

    # 删除包含缺失值的行（如果缺失值比例较低）
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
        print(f"删除缺失值后的数据形状: {df.shape}")

    # 检查并删除重复行
    duplicates = df.duplicated()
    if duplicates.sum() > 0:
        df = df.drop_duplicates()
        print(f"删除重复行后的数据形状: {df.shape}")

    # 编码类别特征
    categorical_features = ['小区名称', '房屋朝向', '楼层部位']

    encoders = {}
    for feature in categorical_features:
        if feature in df.columns:
            encoder = LabelEncoder()
            df[feature] = encoder.fit_transform(df[feature])
            encoders[feature] = encoder

    # 查看数据统计摘要
    print("\n数据统计摘要:")
    print(df.describe())

    return df, encoders


def add_price_category(df, price_col='总价'):
    """
    根据房价添加价格类别

    参数:
    df: DataFrame, 数据
    price_col: str, 价格列名

    返回:
    df: DataFrame, 添加价格类别后的数据
    category_names: list, 类别名称
    """
    # 计算分位数，用于确定价格区间
    price_quantiles = df[price_col].quantile([0.25, 0.5, 0.75]).tolist()

    # 四个价格区间
    price_ranges = [
        (0, price_quantiles[0]),  # 周转性低价房
        (price_quantiles[0], price_quantiles[1]),  # 适用性中低价房
        (price_quantiles[1], price_quantiles[2]),  # 改善中高价房
        (price_quantiles[2], float('inf'))  # 豪华高价房
    ]

    # 添加价格类别
    def get_price_category(price):
        for i, (low, high) in enumerate(price_ranges):
            if low <= price < high:
                return i
        return len(price_ranges) - 1

    df['价格类别'] = df[price_col].apply(get_price_category)

    # 价格类别分布
    category_counts = df['价格类别'].value_counts().sort_index()
    category_names = ['周转性低价房', '适用性中低价房', '改善中高价房', '豪华高价房']

    print("\n价格类别分布:")
    for i, name in enumerate(category_names):
        if i in category_counts:
            print(f"  {name}: {category_counts[i]} ({category_counts[i] / len(df) * 100:.2f}%)")

    # 添加价格分段详情
    print("\n价格区间详情:")
    for i, (low, high) in enumerate(price_ranges):
        if i == len(price_ranges) - 1:
            print(f"  {category_names[i]}: > {low:.2f}万元")
        else:
            print(f"  {category_names[i]}: {low:.2f}万元 - {high:.2f}万元")

    return df, category_names


def prepare_regression_data(df, target_col='总价', test_size=0.2, random_state=42):
    """
    准备房价回归任务的数据

    参数:
    df: DataFrame, 预处理后的数据
    target_col: str, 目标变量列名
    test_size: float, 测试集比例
    random_state: int, 随机种子

    返回:
    X_train, X_test, y_train, y_test: 训练集和测试集
    feature_names: list, 特征名称
    scaler: StandardScaler, 标准化器
    """
    # 选择特征和目标
    if '小区名称' in df.columns:
        # 排除小区名称作为特征，因为它可能导致过拟合
        X = df.drop([target_col, '小区名称', '价格类别'], axis=1, errors='ignore')
    else:
        X = df.drop([target_col, '价格类别'], axis=1, errors='ignore')

    y = df[target_col]
    feature_names = X.columns.tolist()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\n回归任务数据准备完成:")
    print(f"  训练集大小: {X_train.shape[0]}")
    print(f"  测试集大小: {X_test.shape[0]}")
    print(f"  特征数量: {X_train.shape[1]}")
    print(f"  特征名称: {feature_names}")

    return X_train, X_test, y_train, y_test, feature_names, scaler


def prepare_classification_data(df, target_col='价格类别', test_size=0.2, random_state=42):
    """
    准备房价分类任务的数据

    参数:
    df: DataFrame, 预处理后的数据
    target_col: str, 目标变量列名
    test_size: float, 测试集比例
    random_state: int, 随机种子

    返回:
    X_train, X_test, y_train, y_test: 训练集和测试集
    feature_names: list, 特征名称
    scaler: StandardScaler, 标准化器
    """
    # 选择特征和目标
    if '小区名称' in df.columns:
        # 排除小区名称和总价/单价作为特征
        X = df.drop([target_col, '小区名称', '总价', '单价'], axis=1, errors='ignore')
    else:
        X = df.drop([target_col, '总价', '单价'], axis=1, errors='ignore')

    y = df[target_col]
    feature_names = X.columns.tolist()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\n分类任务数据准备完成:")
    print(f"  训练集大小: {X_train.shape[0]}")
    print(f"  测试集大小: {X_test.shape[0]}")
    print(f"  特征数量: {X_train.shape[1]}")
    print(f"  特征名称: {feature_names}")
    print(f"  类别分布: {np.bincount(y_train)}")

    return X_train, X_test, y_train, y_test, feature_names, scaler