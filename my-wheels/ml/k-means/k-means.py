import numpy as np
import matplotlib.pyplot as plt

##��������(Xi,Yi)����Ҫת��������(�б�)��ʽ
Xn=np.array([2,3,1.9,2.5,4])
Yn=np.array([5,4.8,4,1.8,2.2])

#��ʶ����
sign_n = ['A','B','C','D','E']
sign_k = ['k1','k2']

##�����ѡһ�����ݵ���Ϊ���ӵ�
def select_seed(Xn):
    idx = np.random.choice(range(len(Xn)))
    return idx
    
##�������ݵ㵽���ӵ�ľ���
def cal_dis(Xn,Yn,idx):
    dis_list = []
    for i in range(len(Xn)):       
        d = np.sqrt((Xn[i]-Xn[idx])**2+(Yn[i]-Yn[idx])**2)
        dis_list.append(d)
    return dis_list

##�����ѡ��������ӵ�
def select_seed_other(Xn,Yn,dis_list):
    d_sum = sum(dis_list)
    rom = d_sum * np.random.random()
    idx = 0
    for i in range(len(Xn)):
        rom -= dis_list[i]
        if rom > 0 :
            continue
        else :
            idx = i
    return idx

##ѡȡ�������ӵ�
def select_seed_all(seed_count):
     ##���ӵ�
    Xk = []  ##���ӵ�x���б�
    Yk = []  ##���ӵ�y���б�
    
    idx = 0  ##ѡȡ�����ӵ������
    dis_list = [] ##�����б�
    
    
    ##ѡȡ���ӵ�
    #��Ϊʵ�������٣���һ���ļ���ѡ��ͬһ�����ݣ����Լ�һ���ж�
    idx_list = []
    flag = True
    for i in range(seed_count):
        if i == 0:
             idx = select_seed(Xn)
             dis_list = cal_dis(Xn,Yn,idx)
             Xk.append(Xn[idx])
             Yk.append(Yn[idx])
             idx_list.append(idx)
        else :
            while flag:
                idx = select_seed_other(Xn,Yn,dis_list)
                if idx not in idx_list:
                    flag = False
                else :
                    continue
            dis_list = cal_dis(Xn,Yn,idx)
            Xk.append(Xn[idx])
            Yk.append(Yn[idx])
            idx_list.append(idx)
                
    ##�б�ת������       
    Xk=np.array(Xk)
    Yk=np.array(Yk)

    return Xk,Yk
    

def start_class(Xk,Yk):
    ##���ݵ����
    cls_dict = {}
    ##���ĸ����������������ĸ�����
    for i in range(len(Xn)):
        temp = []
        for j in range(len(Xk)):
            d1 = np.sqrt((Xn[i]-Xk[j])*(Xn[i]-Xk[j])+(Yn[i]-Yk[j])*(Yn[i]-Yk[j]))
            temp.append(d1)
        min_dis=np.min(temp)
        min_inx = temp.index(min_dis)
        cls_dict[sign_n[i]]=sign_k[min_inx]
    #print(cls_dict)
    return cls_dict
    
##���¼������������
def recal_class_point(Xk,Yk,cls_dict):  
    num_k1 = 0  #����k1�����ݵ�ĸ���
    num_k2 = 0  #����k2�����ݵ�ĸ���
    x1 =0       #����k1��x�����
    y1 =0       #����k1��y�����
    x2 =0       #����k2��x�����
    y2 =0       #����k2��y�����

    ##ѭ����ȡ�Ѿ����������
    for d in cls_dict:
        ##��ȡd�����
        kk = cls_dict[d]
        if kk == 'k1':
            #��ȡd�����ݼ��е�����
            idx = sign_n.index(d)
            ##�ۼ�xֵ
            x1 += Xn[idx]
            ##�ۼ�yֵ
            y1 += Yn[idx]
            ##�ۼӷ������
            num_k1 += 1
        else :
            #��ȡd�����ݼ��е�����
            idx = sign_n.index(d)
            ##�ۼ�xֵ
            x2 += Xn[idx]
            ##�ۼ�yֵ
            y2 += Yn[idx]
            ##�ۼӷ������
            num_k2 += 1
    ##��ƽ��ֵ��ȡ�µķ��������
    k1_new_x = x1/num_k1 #�µ�k1��x����
    k1_new_y = y1/num_k1 #�µ�k1��y����

    k2_new_x = x2/num_k2 #�µ�k2��x����
    k2_new_y = y2/num_k2 #�µ�k2��y����

    ##�µķ�������
    Xk=np.array([k1_new_x,k2_new_x])
    Yk=np.array([k1_new_y,k2_new_y])
    return Xk,Yk

def draw_point(Xk,Yk,cls_dict):
    #��������
    plt.figure(figsize=(5,4)) 
    plt.scatter(Xn,Yn,color="green",label="����",linewidth=1)
    plt.scatter(Xk,Yk,color="red",label="����",linewidth=1)
    plt.xticks(range(1,6))
    plt.xlim([1,5])
    plt.ylim([1,6])
    plt.legend()
    for i in range(len(Xn)):
        plt.text(Xn[i],Yn[i],sign_n[i]+":"+cls_dict[sign_n[i]])
        for i in range(len(Xk)):
            plt.text(Xk[i],Yk[i],sign_k[i])
    plt.show()

def draw_point_all_seed(Xk,Yk):
    #��������
    plt.figure(figsize=(5,4)) 
    plt.scatter(Xn,Yn,color="green",label="����",linewidth=1)
    plt.scatter(Xk,Yk,color="red",label="����",linewidth=1)
    plt.xticks(range(1,6))
    plt.xlim([1,5])
    plt.ylim([1,6])
    plt.legend()
    for i in range(len(Xn)):
        plt.text(Xn[i],Yn[i],sign_n[i])
    plt.show()

if __name__ == "__main__":

     ##ѡȡ2�����ӵ�
     Xk,Yk = select_seed_all(2)
     ##�鿴���ӵ�
     draw_point_all_seed(Xk,Yk)
     ##ѭ�����ν��з���
     for i in range(3):
        cls_dict =start_class(Xk,Yk)
        Xk_new,Yk_new =recal_class_point(Xk,Yk,cls_dict)
        Xk=Xk_new
        Yk=Yk_new
        draw_point(Xk,Yk,cls_dict)