"""
对比测试Apriori/FP-growth四种实现（手写/库）
数据：T100~T500
最小支持度0.3，最小置信度0.7
输出所有频繁1,2,3项集及关联规则
"""
from apriori_handwritten import apriori, generate_rules
from fpgrowth_handwritten import fp_growth, generate_rules_fp
from apriori_library import create_one_hot as create_one_hot_ap, apriori as apriori_lib, association_rules as association_rules_ap
from fpgrowth_library import create_one_hot as create_one_hot_fp, fpgrowth as fpgrowth_lib, association_rules as association_rules_fp

import pandas as pd

# 测试数据
transactions = [
    ['牛奶', '面包'],
    ['面包', '尿布', '啤酒', '橙汁'],
    ['牛奶', '尿布', '啤酒', '鸡翅'],
    ['面包', '牛奶', '尿布', '啤酒'],
    ['面包', '牛奶', '尿布', '鸡翅']
]

min_support = 0.3
min_conf = 0.7

def print_freq_itemsets_ap(L, support_data):
    for k in [1,2,3]:
        print(f"频繁{k}项集:")
        for item in L[k-1]:
            print(f"  {set(item)} 支持度: {support_data[item]:.2f}")
        print()

def print_rules_ap(rules):
    print("关联规则:")
    for pre, post, conf in rules:
        print(f"  {set(pre)} => {set(post)}, 置信度: {conf:.2f}")
    print()

def print_freq_itemsets_lib(frequent_itemsets):
    for k in [1,2,3]:
        print(f"频繁{k}项集:")
        sub = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x)==k)]
        for _, row in sub.iterrows():
            print(f"  {set(row['itemsets'])} 支持度: {row['support']:.2f}")
        print()

def print_rules_lib(rules):
    print("关联规则:")
    for _, row in rules.iterrows():
        print(f"  {set(row['antecedents'])} => {set(row['consequents'])}, 置信度: {row['confidence']:.2f}")
    print()

if __name__ == '__main__':
    print("===== 1. Apriori 手写版 =====")
    L, support_data = apriori(transactions, min_support=min_support)
    print_freq_itemsets_ap(L, support_data)
    rules = generate_rules(L, support_data, min_confidence=min_conf)
    print_rules_ap(rules)

    print("===== 2. Apriori 库函数版 =====")
    df_ap = create_one_hot_ap(transactions)
    frequent_itemsets = apriori_lib(df_ap, min_support=min_support, use_colnames=True)
    print_freq_itemsets_lib(frequent_itemsets)
    rules = association_rules_ap(frequent_itemsets, metric='confidence', min_threshold=min_conf)
    print_rules_lib(rules)

    print("===== 3. FP-growth 手写版 =====")
    freq_items, support_data = fp_growth(transactions, min_support=min_support)
    for k in [1,2,3]:
        print(f"频繁{k}项集:")
        for item, support in freq_items:
            if len(item)==k:
                print(f"  {set(item)} 支持度: {support:.2f}")
        print()
    rules = generate_rules_fp(freq_items, support_data, min_confidence=min_conf)
    print_rules_ap(rules)

    print("===== 4. FP-growth 库函数版 =====")
    df_fp = create_one_hot_fp(transactions)
    frequent_itemsets = fpgrowth_lib(df_fp, min_support=min_support, use_colnames=True)
    print_freq_itemsets_lib(frequent_itemsets)
    rules = association_rules_fp(frequent_itemsets, metric='confidence', min_threshold=min_conf)
    print_rules_lib(rules)
