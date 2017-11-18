import numpy as np

data = np.array([[1, 2, 0], [3, 4, -1],[5,6, -2],[7,8,-3]])

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
        while(idx < n):
            
            x = data[idx:idx+3, :-1]
            y = data[idx:idx+3,-1]
            g = np.dot(x, theta) - y
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