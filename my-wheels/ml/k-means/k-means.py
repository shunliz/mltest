import numpy as np
import matplotlib.pyplot as plt

##样本数据(Xi,Yi)，需要转换成数组(列表)形式
Xn=np.array([2,3,1.9,2.5,4])
Yn=np.array([5,4.8,4,1.8,2.2])

#标识符号
sign_n = ['A','B','C','D','E']
sign_k = ['k1','k2']

##随机挑选一个数据点作为种子点
def select_seed(Xn):
    idx = np.random.choice(range(len(Xn)))
    return idx
    
##计算数据点到种子点的距离
def cal_dis(Xn,Yn,idx):
    dis_list = []
    for i in range(len(Xn)):       
        d = np.sqrt((Xn[i]-Xn[idx])**2+(Yn[i]-Yn[idx])**2)
        dis_list.append(d)
    return dis_list

##随机挑选另外的种子点
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

##选取所有种子点
def select_seed_all(seed_count):
     ##种子点
    Xk = []  ##种子点x轴列表
    Yk = []  ##种子点y轴列表
    
    idx = 0  ##选取的种子点的索引
    dis_list = [] ##距离列表
    
    
    ##选取种子点
    #因为实验数据少，有一定的几率选到同一个数据，所以加一个判断
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
                
    ##列表转成数组       
    Xk=np.array(Xk)
    Yk=np.array(Yk)

    return Xk,Yk
    

def start_class(Xk,Yk):
    ##数据点分类
    cls_dict = {}
    ##离哪个分类点最近，属于哪个分类
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
    
##重新计算分类的坐标点
def recal_class_point(Xk,Yk,cls_dict):  
    num_k1 = 0  #属于k1的数据点的个数
    num_k2 = 0  #属于k2的数据点的个数
    x1 =0       #属于k1的x坐标和
    y1 =0       #属于k1的y坐标和
    x2 =0       #属于k2的x坐标和
    y2 =0       #属于k2的y坐标和

    ##循环读取已经分类的数据
    for d in cls_dict:
        ##读取d的类别
        kk = cls_dict[d]
        if kk == 'k1':
            #读取d在数据集中的索引
            idx = sign_n.index(d)
            ##累加x值
            x1 += Xn[idx]
            ##累加y值
            y1 += Yn[idx]
            ##累加分类个数
            num_k1 += 1
        else :
            #读取d在数据集中的索引
            idx = sign_n.index(d)
            ##累加x值
            x2 += Xn[idx]
            ##累加y值
            y2 += Yn[idx]
            ##累加分类个数
            num_k2 += 1
    ##求平均值获取新的分类坐标点
    k1_new_x = x1/num_k1 #新的k1的x坐标
    k1_new_y = y1/num_k1 #新的k1的y坐标

    k2_new_x = x2/num_k2 #新的k2的x坐标
    k2_new_y = y2/num_k2 #新的k2的y坐标

    ##新的分类数组
    Xk=np.array([k1_new_x,k2_new_x])
    Yk=np.array([k1_new_y,k2_new_y])
    return Xk,Yk

def draw_point(Xk,Yk,cls_dict):
    #画样本点
    plt.figure(figsize=(5,4)) 
    plt.scatter(Xn,Yn,color="green",label="数据",linewidth=1)
    plt.scatter(Xk,Yk,color="red",label="分类",linewidth=1)
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
    #画样本点
    plt.figure(figsize=(5,4)) 
    plt.scatter(Xn,Yn,color="green",label="数据",linewidth=1)
    plt.scatter(Xk,Yk,color="red",label="分类",linewidth=1)
    plt.xticks(range(1,6))
    plt.xlim([1,5])
    plt.ylim([1,6])
    plt.legend()
    for i in range(len(Xn)):
        plt.text(Xn[i],Yn[i],sign_n[i])
    plt.show()

if __name__ == "__main__":

     ##选取2个种子点
     Xk,Yk = select_seed_all(2)
     ##查看种子点
     draw_point_all_seed(Xk,Yk)
     ##循环三次进行分类
     for i in range(3):
        cls_dict =start_class(Xk,Yk)
        Xk_new,Yk_new =recal_class_point(Xk,Yk,cls_dict)
        Xk=Xk_new
        Yk=Yk_new
        draw_point(Xk,Yk,cls_dict)