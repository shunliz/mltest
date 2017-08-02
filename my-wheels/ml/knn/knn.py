#encoding:utf-8
from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    #���ء����顱�����������shape[1]���ص��������������
    dataSetSize = dataSet.shape[0]
    #���������顱������õ��µ�����
    diffMat = tile(inX,(dataSetSize,1))- dataSet
    #��ƽ��
    sqDiffMat = diffMat **2
    #��ͣ����ص���һά����
    sqDistances = sqDiffMat.sum(axis=1)
    #�����������Ե㵽���������ľ���
    distances = sqDistances **0.5
    #���򣬷���ֵ��ԭ�����С����������±�ֵ
    sortedDistIndicies = distances.argsort()
    #����һ���յ��ֵ�
    classCount = {}
    for i in range(k):
        #���ؾ��������k��������Ӧ�ı�ǩֵ
        voteIlabel = labels[sortedDistIndicies[i]]
        #��ŵ��ֵ���
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    #���� classCount.iteritems() �����ֵ�� key��������Ĺؼ��� True������
    sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse = True)
    #���ؾ�����С�ĵ��Ӧ�ı�ǩ
   return sortedClassCount[0][0]

'''
import kNN  
from numpy import *   
  
dataSet, labels = kNN.createDataSet()  
  
testX = array([1.2, 1.0])  
k = 3  
outputLabel = kNN.kNNClassify(testX, dataSet, labels, 3)  
print "Your input is:", testX, "and classified to class: ", outputLabel  
  
testX = array([0.1, 0.3])  
outputLabel = kNN.kNNClassify(testX, dataSet, labels, 3)  
print "Your input is:", testX, "and classified to class: ", outputLabel  
'''