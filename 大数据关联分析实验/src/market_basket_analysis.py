"""
超市购物篮数据关联规则分析（Apriori/FP-growth库版）
步骤：
1. 数据探索与可视化
2. 数据清洗与预处理
3. 按星期分析交易量
4. 数据格式转换
5. Apriori算法频繁项集与关联规则分析
6. FP-growth算法频繁项集与关联规则分析
7. 结果业务解读与建议
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

# 1. 数据加载与探索
file_path = '../data/BreadBasket_DMS.csv'
df = pd.read_csv(file_path)

print('数据集基本信息:')
print(df.info())
print('\n前5行:')
print(df.head())

# 2. 数据预处理
# 去除缺失值、未购买项（Item为NONE）、重复项
print('\n缺失值统计:')
print(df.isnull().sum())
df = df.dropna()
df = df[df['Item'].str.upper() != 'NONE']
df = df.drop_duplicates()

# 3. 热销商品分析
item_counts = df['Item'].value_counts()
print('\n热销商品Top15:')
print(item_counts.head(15))
plt.figure(figsize=(10,6))
item_counts.head(15).plot(kind='bar')
plt.title('Top 15 热销商品')
plt.ylabel('销量')
plt.xlabel('商品')
plt.tight_layout()
plt.show()

# 4. 按星期分析交易量
# 增加星期字段
import datetime
df['Date'] = pd.to_datetime(df['Date'])
df['Weekday'] = df['Date'].dt.day_name()
weekday_counts = df.groupby('Weekday')['Transaction'].nunique().reindex([
    'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
print('\n各星期交易量:')
print(weekday_counts)
plt.figure(figsize=(8,5))
weekday_counts.plot(kind='bar')
plt.title('各星期交易量')
plt.ylabel('交易数')
plt.xlabel('星期')
plt.tight_layout()
plt.show()

# 5. 数据格式转换（Transaction列表）
transactions = df.groupby('Transaction')['Item'].apply(list).tolist()

# 6. One-hot编码
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

# 7. Apriori算法分析
print('\n===== Apriori 频繁项集（支持度≥0.05） =====')
frequent_itemsets = apriori(df_onehot, min_support=0.05, use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
print(frequent_itemsets.head(20))

print('\n===== Apriori 关联规则（置信度≥0.5） =====')
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
rules = rules.sort_values(by='confidence', ascending=False)
print(rules[['antecedents','consequents','support','confidence','lift']].head(20))

print('\n===== Apriori 关联规则（提升度>1，真实联系） =====')
rules_lift = rules[rules['lift'] > 1].sort_values(by='lift', ascending=False)
print(rules_lift[['antecedents','consequents','support','confidence','lift']].head(20))

# 8. FP-growth算法分析
print('\n===== FP-growth 频繁项集（支持度≥0.05） =====')
fp_itemsets = fpgrowth(df_onehot, min_support=0.05, use_colnames=True)
fp_itemsets = fp_itemsets.sort_values(by='support', ascending=False)
print(fp_itemsets.head(20))

print('\n===== FP-growth 关联规则（置信度≥0.5） =====')
fp_rules = association_rules(fp_itemsets, metric='confidence', min_threshold=0.5)
fp_rules = fp_rules.sort_values(by='confidence', ascending=False)
print(fp_rules[['antecedents','consequents','support','confidence','lift']].head(20))

# 9. 业务分析与建议（示例输出）
print('\n【业务分析建议】')
print('1. 热销商品主要为咖啡、面包等，可重点备货。')
print('2. 频繁项集和高置信度、高提升度规则显示：部分商品常被一起购买（如Coffee与Bread），可考虑捆绑销售。')
print('3. 周末交易量显著高于工作日，可在周末增加促销活动。')
print('4. FP-growth与Apriori结果高度一致，FP-growth更适合大数据场景。')
