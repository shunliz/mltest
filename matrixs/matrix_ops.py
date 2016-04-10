import numpy as np

#define one dim array
a1=np.array([1,2,3],dtype=int)
print a1
#define two dim array
a2=np.array([[1,2,3],[2,3,4]])
print a2

#generate matrix and fill with zero
b1=np.zeros((2,3))
print b1

b2=np.identity(4)
print b2

b3=np.eye(3,M=None,k=0)  
print b3

c1=np.arange(2,3,0.1)  
print c1

c2=np.linspace(1,4,10)  
print c2

a = np.array([[1,2,3], [4,5,6], [7,8,9]])

e1=np.random.rand(3,2)  
print e1

xx=np.roll(a,2)  
print xx


#array features
print a.flags

print a.shape

print a.ndim

print a.size

print a.itemsize

print a.dtype

print a.T

print a.trace()

print np.linalg.det(a)
print np.linalg.norm(a,ord=None)
print np.linalg.eig(a)
print np.linalg.cond(a, p=None)
print np.linalg.inv(a)





