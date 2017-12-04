#-*-coding:utf-8-*-  
# LANG=en_US.UTF-8  
# ���ر�Ҷ˹  
# �ļ�����native_bayes.py  
  
import sys  
import math  
  
# ���� 3 ���б�������������Ƕ�Ӧ�ģ��磺  
#   ��ĳ��Ԫ�� X ������ֵΪ (data_x1[0], data_x2[0]) ʱ�������Ϊ data_y[0]  
# ��ʵ����Ϊ��ʵ�֣�ѵ�����ݱ�:  
#    -------------------------------------------------------------------  
#    |     1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  |  
#    -------------------------------------------------------------------  
#    | x1  1   1   1   1   1   2   2   2   2   2   3   3   3   3   3   |  
#    -------------------------------------------------------------------  
#    | x2  S   M   M   S   S   S   M   M   L   L   L   M   M   L   L   |  
#    -------------------------------------------------------------------  
#    | y   -1  -1  1   1   -1  -1  -1  1   1   1   1   1   1   -1  -1  |  
#   -------------------------------------------------------------------  
  
# ����ֵ x1  
data_x1 = [  
        1, 1, 1, 1, 1,  
        2, 2, 2, 2, 2,  
        3, 3, 3, 3, 3  
    ]  
  
# ����ֵ x2  
data_x2 = [  
        'S', 'M', 'M', 'S', 'S',  
        'S', 'M', 'M', 'L', 'L',  
        'L', 'M', 'M', 'L', 'L'  
    ]  
  
# ��� y  
data_y = [  
        -1, -1, 1, 1, -1,  
        -1, -1, 1, 1, 1,  
        1, 1, 1, 1, -1  
    ]  
  
  
# ����������͵����������������ֵ�  
def get_type_sum( type_list ):  
    # type_dict �������  
    #   key�����ֵ  
    #   value���������  
    # len(type_dict)��������  
    type_dict = {}  
    tmp_item = ''  
  
    # �������ͣ�ͳ��ÿ�����͵����������䱣�浽�ֵ���  
    for item in type_list:  
        item = str(item)  
        if tmp_item != item:  
            if item in type_dict.keys():  
                type_dict[item] += 1.0  
            else:  
                type_dict[item] = 1.0  
                tmp_item = item  
        else:  
            type_dict[item] += 1.0  
  
    return type_dict  
  
  
# ���� P(Xj|Yi)  
def get_Pxjyi( type_list, type_dict, *data ):  
    Pxjyi_dict = {}  
    tmp_type = ''  
    tmp_key = ''  
  
    # ����ԭʼ���ݣ�ͳ��ÿ��������ĳ�������³��ֵ�Ƶ��  
    for num in xrange( len(data[0]) ):  
        x_num = 1  
        for each_data in data:  
            key = 'x%d=%s|y=%s' % ( x_num, str(each_data[num]), type_list[num] )  
            if tmp_key != key:  
                if key in Pxjyi_dict.keys():  
                    Pxjyi_dict[key] += 1  
                else:  
                    Pxjyi_dict[key] = 1  
                    tmp_key = key  
            else:  
                Pxjyi_dict[key] += 1  
            x_num += 1  
  
    for key in Pxjyi_dict:  
        Pxjyi_dict[key] = '%.4f' % ( Pxjyi_dict[key] / type_dict[key.split('y=')[1]] )  
  
    return Pxjyi_dict  
  
  
# ���� P(Yi), ���������ֵ�  
def get_Pyi( type_list, type_dict ):  
    # �� type_dict ��ֵ��ͳ�����͵�������Ϊĳ������ռ�����͵ı���  
    for key in type_dict:  
        type_dict[key] = type_dict[key] / len( type_list )  
  
    return type_dict  
  
  
# �ж�Ŀ�����ݵ���������  
def get_result_type( type_dict, Pxjyi_dict, target_x ):  
    max_probability = 0.0  
    result_type = ''  
  
    # ���� target_x= (2, 'S')  
    # ����������Ƿֱ���� P(Y=1)*P(X1=2|Y=1)*P(X2=S|Y=1) �� P(Y=-1)*P(X1=2|Y=-1)*P(X2=S|Y=-1)  
    # Ȼ������ĸ�ֵ���ж� target_x �����ĸ�����   
    for key in type_dict:  
        for num in xrange( len(target_x) ):  
            value = target_x[num]  
            num += 1  
            Pxjyi_key = 'x%d=%s|y=%s' % ( num, value, key )  
            type_dict[key] = float(type_dict[key]) * float(Pxjyi_dict[Pxjyi_key])  
  
        if type_dict[key] > max_probability:  
            max_probability = type_dict[key]  
            result_type = key  
  
    return result_type, max_probability  
  
  
type_dict = get_type_sum( data_y )  
Pxjyi_dict = get_Pxjyi( data_y, type_dict, data_x1, data_x2 )  
type_dict = get_Pyi( data_y, type_dict )  
target_x= (2, 'S')  
result_type, max_probability = get_result_type( type_dict, Pxjyi_dict, target_x )  
print 'target type is', result_type  