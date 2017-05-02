#!/usr/bin/python
#  -*- coding:utf-8 -*-

import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
 
# 加载数据
# 备用地址: http://cdn.powerxing.com/files/lr-binary.csv
df = pd.read_csv("http://www.ats.ucla.edu/stat/data/binary.csv")
 
# 浏览数据集
print df.head()
#    admit  gre   gpa  rank
# 0      0  380  3.61     3
# 1      1  660  3.67     3
# 2      1  800  4.00     1
# 3      1  640  3.19     4
# 4      0  520  2.93     4
 
# 重命名'rank'列，因为dataframe中有个方法名也为'rank'
df.columns = ["admit", "gre", "gpa", "prestige"]
print df.columns
# array([admit, gre, gpa, prestige], dtype=object)

# summarize the data
print df.describe()
#             admit         gre         gpa   prestige
# count  400.000000  400.000000  400.000000  400.00000
# mean     0.317500  587.700000    3.389900    2.48500
# std      0.466087  115.516536    0.380567    0.94446
# min      0.000000  220.000000    2.260000    1.00000
# 25%      0.000000  520.000000    3.130000    2.00000
# 50%      0.000000  580.000000    3.395000    2.00000
# 75%      1.000000  660.000000    3.670000    3.00000
# max      1.000000  800.000000    4.000000    4.00000
 
# 查看每一列的标准差
print df.std()
# admit      0.466087
# gre      115.516536
# gpa        0.380567
# prestige   0.944460
 
# 频率表，表示prestige与admin的值相应的数量关系
print pd.crosstab(df['admit'], df['prestige'], rownames=['admit'])
# prestige   1   2   3   4
# admit                   
# 0         28  97  93  55
# 1         33  54  28  12
 
# plot all of the columns
df.hist()
pl.show()

