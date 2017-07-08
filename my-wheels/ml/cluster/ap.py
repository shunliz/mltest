from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as plt
'''
��һ�������ɲ�������
    1.����ʵ������Ϊcenters�Ĳ�������300����
    2.Xn�ǰ���150��(x,y)��Ķ�ά����
    3.labels_trueΪ���Ӧ����������ǩ
'''

def init_sample():
    ## ���ɵĲ������ݵ����ĵ�
    centers = [[1, 1], [-1, -1], [1, -1]]
    ##��������
    Xn, labels_true = make_blobs(n_samples=150, centers=centers, cluster_std=0.5,
                            random_state=0)
    #3���ݵĳ��ȣ��������ݵ�ĸ���
    dataLen = len(Xn)

    return Xn,dataLen

'''
�ڶ������������ƶȾ���
'''
def cal_simi(Xn):
    ##������ݼ������ƶȾ��������Ƕ�ά����
    simi = []
    for m in Xn:
        ##ÿ���������������ֵ����ƶ��б��������е�һ��
        temp = []
        for n in Xn:
            ##���ø���ŷʽ����������ƶ�
            s =-np.sqrt((m[0]-n[0])**2 + (m[1]-n[1])**2)
            temp.append(s)
        simi.append(temp)

    ##���òο��ȣ����Խ��ߵ�ֵ��һ��Ϊ��Сֵ������ֵ
    #p = np.min(simi)   ##11������
    #p = np.max(simi)  ##14������
    p = np.median(simi)  ##5������
    for i in range(dataLen):
        simi[i][i] = p
    return simi

'''
�����������������Ⱦ��󣬼�R
       ��ʽ1��r(n+1) =s(n)-(s(n)+a(n))-->��д��������μ���ͼ��ʽ
       ��ʽ2��r(n+1)=(1-��)*r(n+1)+��*r(n)
'''

##��ʼ��R����A����
def init_R(dataLen):
    R = [[0]*dataLen for j in range(dataLen)] 
    return R

def init_A(dataLen):
    A = [[0]*dataLen for j in range(dataLen)]
    return A

##��������R����
def iter_update_R(dataLen,R,A,simi):
    old_r = 0 ##����ǰ��ĳ��rֵ
    lam = 0.5 ##����ϵ��,�����㷨����
    ##��ѭ������R����
    for i in range(dataLen):
        for k in range(dataLen):
            old_r = R[i][k]
            if i != k:
                max1 = A[i][0] + R[i][0]  ##ע���ʼֵ������
                for j in range(dataLen):
                    if j != k:
                        if A[i][j] + R[i][j] > max1 :
                            max1 = A[i][j] + R[i][j]
                ##���º��R[i][k]ֵ
                R[i][k] = simi[i][k] - max1
                ##��������ϵ�����¸���
                R[i][k] = (1-lam)*R[i][k] +lam*old_r
            else:
                max2 = simi[i][0] ##ע���ʼֵ������
                for j in range(dataLen):
                    if j != k:
                        if simi[i][j] > max2:
                            max2 = simi[i][j]
                ##���º��R[i][k]ֵ
                R[i][k] = simi[i][k] - max2
                ##��������ϵ�����¸���
                R[i][k] = (1-lam)*R[i][k] +lam*old_r
    print("max_r:"+str(np.max(R)))
    #print(np.min(R))
    return R
'''
    ���Ĳ�����������Ⱦ��󣬼�A
'''
##��������A����
def iter_update_A(dataLen,R,A):
    old_a = 0 ##����ǰ��ĳ��aֵ
    lam = 0.5 ##����ϵ��,�����㷨����
    ##��ѭ������A����
    for i in range(dataLen):
        for k in range(dataLen):
            old_a = A[i][k]
            if i ==k :
                max3 = R[0][k] ##ע���ʼֵ������
                for j in range(dataLen):
                    if j != k:
                        if R[j][k] > 0:
                            max3 += R[j][k]
                        else :
                            max3 += 0
                A[i][k] = max3
                ##��������ϵ������Aֵ
                A[i][k] = (1-lam)*A[i][k] +lam*old_a
            else :
                max4 = R[0][k] ##ע���ʼֵ������
                for j in range(dataLen):
                    ##��ͼ��ʽ�е�i!=k ����Ͳ���
                    if j != k and j != i:
                        if R[j][k] > 0:
                            max4 += R[j][k]
                        else :
                            max4 += 0

                ##��ͼ��ʽ�е�min����
                if R[k][k] + max4 > 0:
                    A[i][k] = 0
                else :
                    A[i][k] = R[k][k] + max4
                    
                ##��������ϵ������Aֵ
                A[i][k] = (1-lam)*A[i][k] +lam*old_a
    print("max_a:"+str(np.max(A)))
    #print(np.min(A))
    return A

'''
   ��5���������������
'''

##�����������
def cal_cls_center(dataLen,simi,R,A):
    ##���о��࣬���ϵ���ֱ��Ԥ��ĵ������������ж�comp_cnt�κ�������Ĳ��ٱ仯
    max_iter = 100    ##����������
    curr_iter = 0     ##��ǰ��������
    max_comp = 30     ##���Ƚϴ���
    curr_comp = 0     ##��ǰ�Ƚϴ���
    class_cen = []    ##���������б��洢�������ݵ���Xn�е�����
    while True:
        ##����R����
        R = iter_update_R(dataLen,R,A,simi)
        ##����A����
        A = iter_update_A(dataLen,R,A)
        ##��ʼ�����������
        for k in range(dataLen):
            if R[k][k] +A[k][k] > 0:
                if k not in class_cen:
                    class_cen.append(k)
                else:
                    curr_comp += 1
        curr_iter += 1
        print(curr_iter)
        if curr_iter >= max_iter or curr_comp > max_comp :
            break
    return class_cen
  
   
if __name__=='__main__':
    ##��ʼ������
    Xn,dataLen = init_sample()
    ##��ʼ��R��A����
    R = init_R(dataLen)
    A = init_A(dataLen)
    ##�������ƶ�
    simi = cal_simi(Xn)   
    ##�����������
    class_cen = cal_cls_center(dataLen,simi,R,A)
    #for i in class_cen:
    #    print(str(i)+":"+str(Xn[i]))
    #print(class_cen)

    ##���ݾ������Ļ�������
    c_list = []
    for m in Xn:
        temp = []
        for j in class_cen:
            n = Xn[j]
            d = -np.sqrt((m[0]-n[0])**2 + (m[1]-n[1])**2)
            temp.append(d)
        ##�����ǵڼ���������Ϊ�������Ľ��з����ʶ
        c = class_cen[temp.index(np.max(temp))]
        c_list.append(c)
    ##��ͼ
    colors = ['red','blue','black','green','yellow']
    plt.figure(figsize=(8,6))
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    for i in range(dataLen):
        d1 = Xn[i]
        d2 = Xn[c_list[i]]
        c = class_cen.index(c_list[i])
        plt.plot([d2[0],d1[0]],[d2[1],d1[1]],color=colors[c],linewidth=1)
        #if i == c_list[i] :
        #    plt.scatter(d1[0],d1[1],color=colors[c],linewidth=3)
        #else :
        #    plt.scatter(d1[0],d1[1],color=colors[c],linewidth=1)
    plt.show()