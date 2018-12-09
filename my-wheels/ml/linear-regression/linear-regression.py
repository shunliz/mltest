import numpy as np


class LinearRegression(object):
    def __init__(self, alpha=None, lamda=None, batch=None):
        self.theta = None
        self.alpha = alpha
        self.lamda = lamda
        self.batch = batch
        
        if not self.alpha:
            self.alpha = 0.2
        if not self.lamda:
            self.lamda = 0.3
        if not self.batch:
            self.batch = 3
    
    def fit(self, fx, fy):
        n = len(fx[0])
        self.theta = np.zeros(n)
        idx=0;
        times = 0;
        stop = False
        "# one data(if self.batch=1) every theta update, stochastic gradient."
        "# m data every theta update, mini-batch"
        while(idx < len(fx)-1 and not stop):
            times +=1
            x = fx[idx:idx+self.batch]
            y = fy[idx:idx+self.batch]
            g = np.dot(x, self.theta) - y
            """
            def logistic(x):
               return 1/(1+np.exp(x))
            g = logistic(np.dot(x, theta)) -y
            """
            "# lamda*theta, should be the L2 norm"
            "#, lamda, shouble the L1 norm"
            "#, r*lamda + (1-r)lamda*theta, should be the elastic net"
            theta_ = self.theta
            self.theta = self.theta - self.alpha*np.dot(g,x)+self.lamda*self.theta
            # theta = theta - alpha*g*x + lamda
            # theta = theta - alpha*g*x+(r*lamda+(1-r)lamda*theta)
            idx+= self.batch
            diff = theta_ - self.theta
            stop = True if diff.all() <0.00001 and diff.all() >-0.00001 else False
            
        print times, self.theta
    
    def predict(self, x):
        return np.dot(x, self.theta)



data = np.array([[1, 2, 0], [3, 4, -1],[5,6, -2],[7,8,-3]])
x = data[:,:-1]
y = data[:,-1]

li = LinearRegression()
li.fit(x,y)
x_test = np.array([[4,4],[3,3]])
y_predict = li.predict(x_test)

print y_predict


""""

def regression(data, alpha, lamda):
    n = len(data[0]) -1
    theta = np.zeros(n)
    for times in range(100):
        "# one data every theta update, stochastic gradient."
        "# m data every theta update, mini-batch"
        for d in data:
            x = d[:-1]
            y = d[-1]
            g = np.dot(x, theta) - y
            "# lamda*theta, should be the L2 norm"
            "#, lamda, shouble the L1 norm"
            "#, r*lamda + (1-r)lamda*theta, should be the elastic net"
            theta = theta - alpha*np.dot(g,x)+lamda*theta
            # theta = theta - alpha*g*x + lamda
            # theta = theta - alpha*g*x+(r*lamda+(1-r)lamda*theta)
        print times, theta
    return theta

regression(data, 0.2, 0.3)

def regression2(data, alpha, lamda):
    n = len(data[0]) -1
    theta = np.zeros(n)
    for times in range(100):
        "# one data every theta update, stochastic gradient."
        "# m data every theta update, mini-batch"
        idx=0;
        while(idx < len(data)-1):
            x = data[idx:idx+3, :-1]
            y = data[idx:idx+3,-1]
            g = np.dot(x, theta) - y

            def logistic(x):
               return 1/(1+np.exp(x))
            g = logistic(np.dot(x, theta)) -y

            "# lamda*theta, should be the L2 norm"
            "#, lamda, shouble the L1 norm"
            "#, r*lamda + (1-r)lamda*theta, should be the elastic net"
            theta = theta - alpha*np.dot(g,x)+lamda*theta
            # theta = theta - alpha*g*x + lamda
            # theta = theta - alpha*g*x+(r*lamda+(1-r)lamda*theta)
            idx+=3
        print times, theta
    return theta

regression2(data, 0.2, 0.3)
"""