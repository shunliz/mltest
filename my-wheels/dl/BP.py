#coding=utf-8   
from numpy import *  
#from math import *  
from numpy.distutils.core import numpy_cmdclass  
f=open( r'test')  
input=[]  
#数据预处理，把文件数据转换  
for each in f:  
    input.append(each.strip().split())  
n,m,p,t=input[0]  
sample=input[1:int(n)+1]  
w_in_hidden=input[int(n)+1:int(n)+6]  
w_hidden_out=input[int(n)+6:]  
feature=[]#特征矩阵  
lable=[]#标记  
for each in sample:  
    feature.append(each[:-1])  
    lable.append(each[-1])  
#将list转化成矩阵  
feature=mat(feature)  
lable=mat(lable)  
w_in_hidden=mat(w_in_hidden)#隐藏层与输入层的权值矩阵  
w_hidden_out=mat(w_hidden_out)#隐藏层与输出层的权值矩阵  
#逆置  
feature=feature.T  
zero=mat(ones(feature.shape[0]))  
feature=row_stack((zero,feature))  
#将第0行加入矩阵，属矩阵拼接问题  
feature=feature.astype(dtype=float)  
#生成新的矩阵，并改变矩阵内部数据类型，以前是str型的  
w_in_hidden=w_in_hidden.astype(dtype=float)  
lable=lable.astype(dtype=float)  
w_hidden_out=w_hidden_out.astype(dtype=float)  
hidden_output=dot(w_in_hidden,feature)  
hidden_output=hidden_output.T  
#此处exp是numpy里面自带的求矩阵指数的函数  
hidden_output=1/(1+exp(-1*hidden_output))  
print hidden_output#隐藏层的输出  
hidden_output=hidden_output.T  
zero=mat(ones(hidden_output.shape[1]))  
hidden_output=row_stack((zero,hidden_output))  
output=dot(w_hidden_out,hidden_output)  
output=output.T  
output=1/(1+exp(-1*output))  
print output#输出层的输出  
#lable原本的值是3,2,1代表的是第一次输出第三个输出单元输出为1，第二次输出第二个输出单元输出为1...  
lable=mat([[0,0,1],[0,1,0],[1,0,0]])  
lable=lable.T  
output=output.tolist()#将矩阵转化回list  
lable=lable.tolist()  
sum=0.0  
#计算误差，其实也可以直接用矩阵计算，问题在于本人没有找到求矩阵对角线和的函数，且做一标记，找到补上  
for i in range (len(output)):  
    for j in range (len(output[0])):  
        sum+=math.log(output[i][j])*-lable[i][j]-math.log(1-output[i][j])*(1-lable[i][j])  
print sum/3 