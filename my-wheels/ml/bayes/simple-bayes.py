#encoding:utf-8

from numpy import *

#�ʱ�������ת������
def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]      #1,����  0,����
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])  #����set����,����һ���ռ�
    for document in dataSet:
        vocabSet = vocabSet | set(document)     #�����������ϵĲ���
    return list(vocabSet)
'''
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)   #����һ������Ԫ�ض�Ϊ0������
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word:%s is not in my Vocabulary" % word
    return returnVec
'''

def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)   #����һ������Ԫ�ض�Ϊ0������
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


#���ر�Ҷ˹������ѵ����
def trainNB0(trainMatrix,trainCategory):  #�������Ϊ�ĵ�����ÿƪ�ĵ�����ǩ�����ɵ�����
    numTrainDocs = len(trainMatrix)      #�ĵ�����ĳ���
    numWords = len(trainMatrix[0])       #��һ���ĵ��ĵ��ʸ���
    pAbusive = sum(trainCategory)/float(numTrainDocs)  #�����ĵ������������ĵ�����
    #p0Num = zeros(numWords);p1Num = zeros(numWords)        #��ʼ���������󣬳���ΪnumWords������ֵΪ0
    p0Num = ones(numWords);p1Num = ones(numWords)        #��ʼ���������󣬳���ΪnumWords������ֵΪ1
    #p0Denom = 0.0;p1Denom = 0.0                         #��ʼ������
    p0Denom = 2.0;p1Denom = 2.0 
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num +=trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num +=trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #p1Vect = p1Num/p1Denom #��ÿ��Ԫ��������
    #p0Vect = p0Num/p0Denom
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

#���ر�Ҷ˹���ຯ��
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)   #Ԫ�����
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()   #�����ĵ�����Ͷ�Ӧ�ı�ǩ
    myVocabList = createVocabList(listOPosts) #��������
    trainMat = []   #����һ���յ��б�
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))  #ʹ�ô����������trainMat�б�
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))  #ѵ������
    testEntry = ['love','my','dalmation']   #�����ĵ��б�
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry)) #��������
    print testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))    #��������
    print testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb)