"""
Apriori算法库函数版（mlxtend）
详细中文注释
"""
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def create_one_hot(transactions):
    """
    将原始事务数据转换为 one-hot 编码的 DataFrame，适用于mlxtend算法。
    参数：
        transactions: 事务数据列表，每个元素是一个商品集合
    返回：
        df: one-hot编码后的pandas DataFrame
    """
    all_items = sorted(set(item for t in transactions for item in t))  # 所有商品去重排序
    encoded = []
    for t in transactions:
        encoded.append([1 if item in t else 0 for item in all_items])  # 有则1，无则0
    return pd.DataFrame(encoded, columns=all_items)

# 用法示例：
# df = create_one_hot(transactions)  # 转换为one-hot
# frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)  # 频繁项集
# rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)  # 关联规则
