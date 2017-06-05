#!/usr/bin/python
#  -*- coding:utf-8 -*-


import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
# read_csv����Ĳ�����csv��������ϵ�·�����˴�csv�ļ�����notebook����Ŀ¼�����CCPPĿ¼��
data = pd.read_csv('.\CCPP\ccpp.csv')

X = data[['AT', 'V', 'AP', 'RH']]
y = data[['PE']]
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1)
ridge.fit(X_train, y_train)

print ridge.coef_
print ridge.intercept_

from sklearn.linear_model import RidgeCV
ridgecv = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100])
ridgecv.fit(X_train, y_train)
ridgecv.alpha_  

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# X is a 10x10 matrix
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
# y is a 10 x 1 vector
y = np.ones(10)

n_alphas = 200
# alphas count is 200, ����10��-10�η���10��-2�η�֮��
alphas = np.logspace(-10, -2, n_alphas)

clf = linear_model.Ridge(fit_intercept=False)
coefs = []
# ѭ��200��
for a in alphas:
    #���ñ���ѭ���ĳ�����
    clf.set_params(alpha=a)
    #���ÿ��alpha��ridge�ع�
    clf.fit(X, y)
    # ��ÿһ��������alpha��Ӧ��theta������
    coefs.append(clf.coef_)
    
ax = plt.gca()

ax.plot(alphas, coefs)
#��alpha��ֵȡ�������ڻ�ͼ
ax.set_xscale('log')
#��תx��Ĵ�С������alpha�Ӵ�С��ʾ
ax.set_xlim(ax.get_xlim()[::-1]) 
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

