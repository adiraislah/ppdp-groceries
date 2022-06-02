# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 09:12:21 2022

@author: Adira Islah (6181801078)
"""

import time
import pandas as pd
from apyori import apriori
from datetime import datetime
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules


# load data groceries
df = pd.read_csv("Groceries.csv")

df.dtypes

#data processing
df['itemDescription'] = df['itemDescription'].str.strip()
df['Member_number'] = df['Member_number'].astype('str')

df = df.reset_index()
basket = df.pivot_table(index=('Member_number','Date'),columns='itemDescription',values='index',aggfunc='count',fill_value=0)

def encode_units3(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units3)

#Applying Apriori
#Frequent item
apriori_start = datetime.now()
frequent_itemset = apriori(basket_sets, min_support=0.001, use_colnames=True)
frequent_itemset_sort = frequent_itemset.sort_values(['support'],ascending=[False,False])
apriori_end = datetime.now()
running_time = apriori_end-apriori_start
print("Apriori Running Time: ", str(running_time))

#Association Rules
apriori_start = datetime.now()
rules_apriori = association_rules(frequent_itemset, metric="lift", min_threshold=0.7)
rules_apriori[ (rules_apriori['lift'] >= 0.7) & (rules_apriori['support'] >= 0.001)]
rules_apriori = rules_apriori.sort_values(['confidence', 'lift'], ascending = [False, False])
apriori_end = datetime.now()
running_time = apriori_end-apriori_start
print("Apriori Running Time: ", str(running_time))


#Applying FP-Growth
#Frequent Item
fpgrowth_start = datetime.now()
fpgrowth_frequent_itemsets = fpgrowth(basket_sets, min_support=0.001, use_colnames=True)
fpgrowth_end = datetime.now()
running_time = fpgrowth_end-fpgrowth_start
print("FP-Growth Running Time: ", str(running_time))

#asc_rules_2 = association_rules(fpgrowth_frequent_itemsets,  metric="confidence", min_threshold=0.2)
fpgrowth_start = datetime.now()
fprules = association_rules(fpgrowth_frequent_itemsets,  metric="lift", min_threshold=0.7)
fprules = fprules[ (fprules['support'] >= 0.001) & (fprules['lift'] > 1) ]
fprules = fprules.sort_values(['support', 'lift'], ascending = [False, False])
apriori_end = datetime.now()
running_time = fpgrowth_end-fpgrowth_start
print("Apriori Running Time: ", str(running_time))