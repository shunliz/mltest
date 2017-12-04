#-*-coding:utf-8-*-  
# LANG=en_US.UTF-8  
# 朴素贝叶斯  
# 文件名：native_bayes.py  
  
import sys  
import math  
  
# 下面 3 个列表的数据其坐标是对应的，如：  
#   当某个元素 X 其特征值为 (data_x1[0], data_x2[0]) 时，其类别为 data_y[0]  
# 其实就是为了实现（训练数据表）:  
#    -------------------------------------------------------------------  
#    |     1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  |  
#    -------------------------------------------------------------------  
#    | x1  1   1   1   1   1   2   2   2   2   2   3   3   3   3   3   |  
#    -------------------------------------------------------------------  
#    | x2  S   M   M   S   S   S   M   M   L   L   L   M   M   L   L   |  
#    -------------------------------------------------------------------  
#    | y   -1  -1  1   1   -1  -1  -1  1   1   1   1   1   1   -1  -1  |  
#   -------------------------------------------------------------------  
  
# 特征值 x1  
data_x1 = [  
        1, 1, 1, 1, 1,  
        2, 2, 2, 2, 2,  
        3, 3, 3, 3, 3  
    ]  
  
# 特征值 x2  
data_x2 = [  
        'S', 'M', 'M', 'S', 'S',  
        'S', 'M', 'M', 'L', 'L',  
        'L', 'M', 'M', 'L', 'L'  
    ]  
  
# 类别 y  
data_y = [  
        -1, -1, 1, 1, -1,  
        -1, -1, 1, 1, 1,  
        1, 1, 1, 1, -1  
    ]  
  
  
# 计算各个类型的总数，返回类型字典  
def get_type_sum( type_list ):  
    # type_dict 保存类别：  
    #   key：类别值  
    #   value：类别数量  
    # len(type_dict)：类别个数  
    type_dict = {}  
    tmp_item = ''  
  
    # 遍历类型，统计每个类型的数量，将其保存到字典中  
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
  
  
# 计算 P(Xj|Yi)  
def get_Pxjyi( type_list, type_dict, *data ):  
    Pxjyi_dict = {}  
    tmp_type = ''  
    tmp_key = ''  
  
    # 遍历原始数据，统计每种数据在某个类型下出现的频率  
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
  
  
# 计算 P(Yi), 返回类型字典  
def get_Pyi( type_list, type_dict ):  
    # 将 type_dict 的值由统计类型的数量改为某个类型占总类型的比例  
    for key in type_dict:  
        type_dict[key] = type_dict[key] / len( type_list )  
  
    return type_dict  
  
  
# 判断目标数据的所属类型  
def get_result_type( type_dict, Pxjyi_dict, target_x ):  
    max_probability = 0.0  
    result_type = ''  
  
    # 这里 target_x= (2, 'S')  
    # 于是下面就是分别计算 P(Y=1)*P(X1=2|Y=1)*P(X2=S|Y=1) 和 P(Y=-1)*P(X1=2|Y=-1)*P(X2=S|Y=-1)  
    # 然后根据哪个值大判断 target_x 属于哪个类型   
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