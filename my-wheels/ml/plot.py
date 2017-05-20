# encoding=utf8
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font  = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)

def runplt():
    plt.figure()
    plt.title(u'披萨价格与直径数据', fontproperties=font)
    plt.xlabel(u'直径（英寸）', fontproperties=font)
    plt.ylabel(u'价格（美元）', fontproperties=font)
    plt.axis([0,25,0,25])
    plt.grid(True)
    return plt

plt = runplt()
X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]
plt.plot(X, y, 'k.')
plt.show()

