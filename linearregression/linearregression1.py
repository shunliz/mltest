#encoding: utf-8

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font  = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)
from sklearn.linear_model import LinearRegression

import numpy as np


def runplt():
    plt.figure()
    plt.title(u'披萨价格与直径数据', fontproperties=font)
    plt.xlabel(u'直径（英寸）', fontproperties=font)
    plt.ylabel(u'价格（美元）', fontproperties=font)
    plt.axis([0,25,0,25])
    plt.grid(True)
    return plt


X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]

model = LinearRegression()
model.fit(X, y)
print(u'预测价格:$%.2f' % model.predict([12])[0])

print(u'残差平方和：%.2f' % np.mean((model.predict(X) - y)**2))

xbar = (6 + 8 + 10 + 14 + 18) / 5
variance = ((6-xbar)**2 + (8-xbar)**2 + (1-xbar)**2 + (14-xbar)**2 + (18-xbar)**2) / 4
print(variance)

plt = runplt()
plt.plot(X, y, 'k.')
X2 = [[0], [10], [14], [25]]
y2 = model.predict(X2)

#增加了成本函数的的预测
y3 = [14.25, 14.25, 14.25,14.25]
y4 = y2*0.5 + 5
model.fit(X[1:-1], y[1:-1])
y5 = model.predict(X2)

#模型的预测效果，R值越接近1说明预测效果越好
X_test = [[8], [9], [11], [16], [12]]
y_test = [[11], [8.5], [15], [18], [11]]
print model.score(X_test, y_test)

yr = model.predict(X)
for idx, x in enumerate(X):
	plt.plot([x,x], [y[idx], yr[idx]], 'r-.')

plt.plot(X ,y, 'k.')
plt.plot(X2,y2, 'g-.')
plt.plot(X2 ,y3, 'r-.')
plt.plot(X2,y4, 'y-.')
plt.plot(X2,y5, 'o-.')

plt.show()


