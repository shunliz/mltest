#encoding: utf-8


from sklearn.linear_model import LinearRegression

X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]

model = LinearRegression()
model.fit(X, y)
print(u'预测价格:$%.2f' % model.predict([12])[0])