"""
Apriori算法纯Python手写实现
详细中文注释
"""
from itertools import combinations, chain

def create_C1(transactions):
    """
    构造候选1项集（C1），即所有商品的集合，每个商品单独成集。
    参数：
        transactions: 事务数据列表，每个元素是一个事务（商品集合）
    返回：
        C1: 候选1项集列表，每个元素是frozenset
    """
    C1 = set()
    for t in transactions:
        for item in t:
            C1.add(frozenset([item]))
    return list(C1)

def scan_D(D, Ck, min_support):
    """
    扫描数据集，计算每个候选项集的支持度，返回满足最小支持度的频繁项集。
    参数：
        D: 事务集，每个事务是set
        Ck: 候选k项集列表
        min_support: 最小支持度（0-1）
    返回：
        ret_list: 满足支持度的频繁项集
        support_data: 所有项集的支持度字典
    """
    ss_cnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                ss_cnt[can] = ss_cnt.get(can, 0) + 1
    num_items = float(len(D))
    ret_list = []
    support_data = {}
    for key in ss_cnt:
        support = ss_cnt[key] / num_items
        if support >= min_support:
            ret_list.append(key)
        support_data[key] = support
    return ret_list, support_data

def apriori_gen(Lk, k):
    """
    根据上一次的频繁(k-1)项集Lk，生成候选k项集Ck
    参数：
        Lk: 上一层频繁项集
        k: 当前项集大小
    返回：
        ret_list: 候选k项集列表
    """
    ret_list = []
    len_Lk = len(Lk)
    for i in range(len_Lk):
        for j in range(i+1, len_Lk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1 == L2:
                ret_list.append(Lk[i] | Lk[j])
    return ret_list

def apriori(transactions, min_support=0.3):
    """
    Apriori主流程，返回所有层的频繁项集和支持度字典
    参数：
        transactions: 原始事务数据
        min_support: 最小支持度
    返回：
        L: 各层频繁项集列表（L[0]是一项集，L[1]是二项集...）
        support_data: 所有项集的支持度字典
    """
    D = list(map(set, transactions))
    C1 = create_C1(D)
    L1, support_data = scan_D(D, C1, min_support)
    L = [L1]
    k = 2
    while len(L[k-2]) > 0:
        Ck = apriori_gen(L[k-2], k)
        Lk, supK = scan_D(D, Ck, min_support)
        support_data.update(supK)
        if not Lk:
            break
        L.append(Lk)
        k += 1
    return L, support_data

def generate_rules(L, support_data, min_confidence=0.7):
    """
    由频繁项集生成满足最小置信度的关联规则（全子集枚举法，完全一致于mlxtend/FP-growth实现）
    参数：
        L: 各层频繁项集列表
        support_data: 支持度字典
        min_confidence: 最小置信度
    返回：
        rules: 满足条件的规则列表（前件, 后件, 置信度）
    """
    rules = []
    # 将所有频繁项集合并成一个集合（跳过1项集）
    freq_sets = set(chain.from_iterable(L[1:]))
    for freq_set in freq_sets:
        for i in range(1, len(freq_set)):
            for antecedent in combinations(freq_set, i):
                antecedent = frozenset(antecedent)
                consequent = freq_set - antecedent
                if len(consequent) == 0:
                    continue
                conf = support_data[freq_set] / support_data[antecedent]
                if conf >= min_confidence:
                    rules.append((antecedent, consequent, conf))
    return rules

def calc_conf(freq_set, H, support_data, rules, min_confidence):
    """
    计算规则的置信度，筛选出满足条件的规则
    """
    pruned_H = []
    for conseq in H:
        conf = support_data[freq_set] / support_data[freq_set - conseq]
        if conf >= min_confidence:
            rules.append((freq_set - conseq, conseq, conf))
            pruned_H.append(conseq)
    return pruned_H

def rules_from_conseq(freq_set, H, support_data, rules, min_confidence):
    """
    递归生成多后件的规则
    """
    m = len(H[0])
    if len(freq_set) > (m + 1):
        Hmp1 = apriori_gen(H, m+1)
        Hmp1 = calc_conf(freq_set, Hmp1, support_data, rules, min_confidence)
        if len(Hmp1) > 1:
            rules_from_conseq(freq_set, Hmp1, support_data, rules, min_confidence)
