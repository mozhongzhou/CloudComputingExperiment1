"""
FP-growth算法纯Python手写实现
详细中文注释
"""
from collections import defaultdict
from itertools import combinations, chain

def create_init_set(transactions):
    """
    将事务数据转换为初始计数字典
    参数：transactions: 事务列表
    返回：dict，每个事务frozenset为key，value为1
    """
    ret_dict = {}
    for trans in transactions:
        key = frozenset(trans)
        ret_dict[key] = ret_dict.get(key, 0) + 1
    return ret_dict

class treeNode:
    def __init__(self, name_value, num_occ, parent_node):
        self.name = name_value
        self.count = num_occ
        self.node_link = None
        self.parent = parent_node
        self.children = {}
    def inc(self, num_occ):
        self.count += num_occ
    def disp(self, ind=1):
        print('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

def update_header(node_to_test, target_node):
    while node_to_test.node_link is not None:
        node_to_test = node_to_test.node_link
    node_to_test.node_link = target_node

def update_tree(items, in_tree, header_table, count):
    if items[0] in in_tree.children:
        in_tree.children[items[0]].inc(count)
    else:
        in_tree.children[items[0]] = treeNode(items[0], count, in_tree)
        if header_table[items[0]][1] is None:
            header_table[items[0]][1] = in_tree.children[items[0]]
        else:
            update_header(header_table[items[0]][1], in_tree.children[items[0]])
    if len(items) > 1:
        update_tree(items[1::], in_tree.children[items[0]], header_table, count)

def create_tree(data_set, min_support=0.3):
    header_table = {}
    for trans in data_set:
        for item in trans:
            header_table[item] = header_table.get(item, 0) + data_set[trans]
    num_items = float(sum(data_set.values()))
    # 只保留满足最小支持度的项
    header_table = {k: v for k, v in header_table.items() if v/num_items >= min_support}
    freq_item_set = set(header_table.keys())
    if len(freq_item_set) == 0:
        return None, None
    for k in header_table:
        header_table[k] = [header_table[k], None]
    ret_tree = treeNode('Null', 1, None)
    for tran_set, count in data_set.items():
        local_d = {}
        for item in tran_set:
            if item in freq_item_set:
                local_d[item] = header_table[item][0]
        if len(local_d) > 0:
            ordered_items = [v[0] for v in sorted(local_d.items(), key=lambda p: (-p[1], p[0]))]
            update_tree(ordered_items, ret_tree, header_table, count)
    return ret_tree, header_table

def ascend_tree(leaf_node, prefix_path):
    if leaf_node.parent is not None:
        prefix_path.append(leaf_node.name)
        ascend_tree(leaf_node.parent, prefix_path)

def find_prefix_path(base_pat, tree_node):
    cond_pats = {}
    while tree_node is not None:
        prefix_path = []
        ascend_tree(tree_node, prefix_path)
        if len(prefix_path) > 1:
            cond_pats[frozenset(prefix_path[1:])] = tree_node.count
        tree_node = tree_node.node_link
    return cond_pats

def mine_tree(in_tree, header_table, min_support, pre_fix, freq_item_list, support_data, num_items):
    # header_table按频率升序遍历
    sorted_items = [v[0] for v in sorted(header_table.items(), key=lambda p: p[1][0])]
    for base_pat in sorted_items:
        new_freq_set = pre_fix | frozenset([base_pat])
        support = header_table[base_pat][0] / num_items
        support_data[new_freq_set] = support
        freq_item_list.append(new_freq_set)
        cond_patt_bases = find_prefix_path(base_pat, header_table[base_pat][1])
        cond_tree, cond_header = create_tree(cond_patt_bases, min_support)
        if cond_header is not None:
            mine_tree(cond_tree, cond_header, min_support, new_freq_set, freq_item_list, support_data, num_items)

def fp_growth(transactions, min_support=0.3):
    """
    FP-growth主流程，返回所有频繁项集和支持度字典
    参数：
        transactions: 原始事务数据
        min_support: 最小支持度
    返回：
        freq_items: 所有频繁项集列表（frozenset, 支持度）
        support_data: 所有项集的支持度字典
    """
    init_set = create_init_set(transactions)
    num_items = float(sum(init_set.values()))
    tree, header_table = create_tree(init_set, min_support)
    if tree is None:
        return [], {}
    freq_item_list = []
    support_data = {}
    mine_tree(tree, header_table, min_support, frozenset(), freq_item_list, support_data, num_items)
    # 只保留支持度>=min_support的项集
    freq_items = [(itemset, support_data[itemset]) for itemset in freq_item_list if support_data[itemset] >= min_support]
    return freq_items, support_data

def generate_rules_fp(freq_items, support_data, min_confidence=0.7):
    """
    由频繁项集生成关联规则（全子集枚举法，保证与Apriori/库一致）
    参数：
        freq_items: 频繁项集及支持度列表
        support_data: 支持度字典
        min_confidence: 最小置信度
    返回：
        rules: 满足条件的规则列表（前件, 后件, 置信度）
    """
    rules = []
    for itemset, _ in freq_items:
        if len(itemset) < 2:
            continue
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                if len(consequent) == 0:
                    continue
                conf = support_data[itemset] / support_data[antecedent]
                if conf >= min_confidence:
                    rules.append((antecedent, consequent, conf))
    return rules
