# encoding=utf8

import numpy as np
from sklearn.datasets import species_distributions
from astropy.wcs.docstrings import spec
from nltk.metrics.scores import precision
from matplotlib.pyplot import axis
print np.__version__

arr = np.arange(10)
print arr

arr2 = np.arange(10,100,10)
print arr2

print np.full((3,3), True, dtype=bool)

print np.zeros((3,3), dtype=bool)

print arr[arr %2 == 1]

#arr[arr %2 ==1] = -1
print np.where(arr %2 == 1, -1, arr)

print arr.reshape(-1,2)

a=np.arange(10).reshape(2,-1)
b=np.repeat(1,10).reshape(2,-1)

print np.concatenate([a,b],axis=0)
print np.concatenate([a,b],axis=None)
#print np.concatenate([a,b],axis=None,out=[4,5])
print np.vstack([a,b])
print np.r_[a,b]

a2=np.arange(10).reshape(2,-1)
b2=np.repeat(1,10).reshape(2,-1)

print np.concatenate([a2,b2],axis=1)
print np.hstack([a2,b2])
print np.c_[a2,b2]

a3=np.array([1,2,3])
print np.r_[np.repeat(a3, 3),np.tile(a3, 3)]

print np.intersect1d(arr, arr[arr %2==0])
print np.setdiff1d(arr, arr[arr %2==0])

m=np.array([1,2,3,2,3,4,3,4,5,6])
n=np.array([7,2,10,2,7,4,9,4,9,8])
print np.where(m == n)

index = np.where((m>=5)&(m<=10))
print m[index]

index2=np.where(np.logical_and(m>=5, m<=10))
print m[index2]

def maxx(x,y):
    if x>=y:
        return x
    else:
        return y
    
pair_max = np.vectorize(maxx, otypes=[float])

l = np.array([5,7,9,8,6,4,5])
k = np.array([6,3,4,8,9,7,1])

print pair_max(l,k)

print np.arange(9).reshape(3,3)[:,[1,0,2]]
print np.arange(9).reshape(3,3)[[1,0,2],:]
print np.arange(9).reshape(3,3)[::-1]
print np.arange(9).reshape(3,3)[:,::-1]
print np.arange(9).reshape(3,3)[:,:-1]
print np.arange(9).reshape(3,3)[:-1]


print np.random.randint(low=5,high=10,size=(5,3))+np.random.random((5,3))

print np.random.uniform(5,10,size=(5,3))

print np.random.random([5,3])
np.set_printoptions(precision=3)
print np.random.random([5,3])[:4]
print np.random.random([5,3])[:4]/1e3


url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names=('sepallength','sepalwidth','petallength','petalwidth','species')
print iris[:3]

iris_1d = np.genfromtxt(url,delimiter=',', dtype=None)
print iris_1d
print np.array([row[4] for row in iris_1d])

iris_2d=np.array([row.tolist()[:4] for row in iris_1d])
print iris_2d


iris_2d2 = np.genfromtxt(url,delimiter=',', dtype='float',usecols=[0,1,2,3])

print iris_2d2[:4]

sepallength = np.genfromtxt(url,delimiter=',', dtype='float',usecols=[0])
mu,med,sd = np.mean(sepallength), np.median(sepallength),np.std(sepallength)
print mu,med,sd

Smax,Smin = sepallength.max(), sepallength.min()
S = (sepallength - Smin)/(Smax - Smin)
print (S)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

print(softmax(sepallength))

print np.percentile(sepallength, q=[5,95])


i, j = np.where(iris_2d)
iris_2d[np.random.choice((i),20), np.random.choice((j),20)] = np.nan
iris_2d[np.random.randint(150,size=20),np.random.randint(4,size=20)] = np.nan
print iris_2d

print "Number of miss values.\n", np.isnan(iris_2d[:,0]).sum()
print "Position of missing values.\n", np.where(np.isnan(iris_2d[:,0]))

condition = (iris_2d2[:,2]>1.5) &(iris_2d2[:,0] <5.0)
print iris_2d2[condition]

any_nan_in_row = np.array([~np.any(np.isnan(row)) for row in iris_2d])
print iris_2d[any_nan_in_row][:5]

url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
print np.corrcoef(iris[:,0], iris[:,2])[0,1]

from scipy.stats.stats import pearsonr
corr, p_value = pearsonr(iris[:,0],iris[:,2])
print corr, p_value

print np.isnan(iris_2d).any()
iris_2d[np.isnan(iris_2d)] = 0


#get the unique values and the counts
#np.unique(species, return_counts=True)
'''
label_map = {1:'small',2:'medium',3:'large',4:np.nan}
petal_length_cat=[label_map[x] for x in petal_length_bin]

print petal_length_cat[:4]
'''

url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')
sepallength = iris_2d[:,0].astype('float')
petallength = iris_2d[:,2].astype('float')
volume = (np.pi * petallength *(sepallength**2))/3

volume = volume[:, np.newaxis]

out = np.hstack([iris_2d, volume])
print out


url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')

species = iris[:,4]
np.random.seed(100)
a = np.array(['Iris-setosa','Iris-versicolor','Iris-virginica'])
species_out = np.random.choice(a,150,p=[0.5,0.25,0.25])
print species_out

np.random.seed(100)
probs = np.r_[np.linspace(0,0.500,num=50), np.linspace(0.501,.750,num=50), np.linspace(.751,1.0,num=50)]
index = np.searchsorted(probs, np.random.random(150))
species_out = species[index]
print np.unique(species_out, return_counts=True)

petal_len_setosa = iris[iris[:,4] == b'Iris-setosa', [2]].astype('float')
print np.unique(np.sort(petal_len_setosa))[-2]

print iris[:,0].argsort()
print iris[iris[:,0].argsort()][:20]

vals,counts = np.unique(iris[:3], return_counts=True)
print vals[np.argmax(counts)]

print np.argmax(iris[:,3].astype('float')>1.0)

np.set_printoptions(precision=2)
np.random.seed(100)
a = np.random.uniform(1,50,20)

print np.clip(a, a_min=10, a_max=30)

print np.where(a<10, 10, np.where(a>30, 30, a))


np.random.seed(100)
a = np.random.uniform(1,50,20)

print a.argsort()

print np.argmax(a)

print a[a.argsort()][-5:]


np.random.seed(100)
arr = np.random.randint(1,11,size=(6,10))

def counts_of_all_values_rowwise(arr2d):
    num_counts_array = [np.unique(row, return_counts=True) for row in arr2d]
    return [[int(b[a==i]) if i in a else 0 for i in np.unique(arr2d)] for a, b in num_counts_array]


arr1 = np.arange(3)
arr2 = np.arange(3,7)
arr3 = np.arange(7,10)

array_of_arrays = np.array([arr1, arr2, arr3])
arr_2d = np.array([a for arr in array_of_arrays for a in arr])


#Input
np.random.seed(101)
arr = np.random.randint(1,4, size=6)
print arr
#Solution
def one_hot_encodings(arr):
    uniqs = np.unique(arr)
    out = np.zeros((arr.shape[0], uniqs.shape[0]))
    for i, k in enumerate(arr):
        out[i,k-1]=1
        return out

print one_hot_encodings(arr)

# Solution 2
(arr[:, None] == np.unique(arr)).view(np.int8)


url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
species_small = np.sort(np.random.choice(species, size=200))
print species_small

print [i for val in np.unique(species_small) for i, grp in enumerate(species_small[species_small == val])]


url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
species_small = np.sort(np.random.choice(species, size=20))
print species_small

#Solution 1
output = [np.argwhere(np.unique(species_small)==s).tolist()[0][0] for val in np.unique(species_small) for s in species_small[species_small==val]]

#Solution
output = []

uniqs = np.unique(species_small)

for val in uniqs:
    for s in species_small[species_small == val]:
        groupid = np.argwhere(uniqs == s).tolist()[0][0]
        output.append(groupid)
        
print (output)



np.random.seed(10)
a = np.random.randint(20, size=10)
print a
print a.argsort()
print a.argsort().argsort()

a = np.random.randint(20, size=[2,5])
print a.ravel().argsort().argsort().reshape(a.shape)

np.random.seed(100)
a = np.random.randint(1,10,[5,3])

print np.amax(a, axis=1)
print np.apply_along_axis(np.max, axis=1, arr=a)

print np.apply_along_axis(lambda x: np.min(x)/np.max(x), arr=a, axis=1)

#在给定的numpy数组中找到重复的条目（从第2个起），并将它们标记为True。第一次出现应该是False。
np.random.seed(100)
a = np.random.randint(0,5,10)
#solution 1
out= np.full(a.shape[0], True)
unique_positions = np.unique(a, return_index=True)[1]
out[unique_positions] = False

#如何找到numpy中的分组平均值ֵ
url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

#Solution
numeric_column = iris[:,1].astype('float')
grouping_column = iris[:,4]

[[group_val, numeric_column[grouping_column == group_val].mean()] for group_val in np.unique(grouping_column)]

output = []
for group_val in np.unique(grouping_column):
    output.append([group_val, numeric_column[grouping_column==group_val].mean()])
    

#问题：从以下URL中导入图像并将其转换为numpy数组
# URL='https://upload.wikimedia.org/wikipedia/commons/8/8b/Denali_Mt_McKinley.jpg'
from io import BytesIO
from PIL import Image
import PIL, requests

URL='https://upload.wikimedia.org/wikipedia/commons/8/8b/Denali_Mt_McKinley.jpg'
response = requests.get(URL)

I = Image.open(BytesIO(response.content))

I = I.resize([150,150])
arr = np.asarray(I)

im = PIL.Image.fromarray(np.uint8(arr))

Image.Image.show(im)

#问题：从一维numpy数组中删除所有nan值
a = np.array([1,2,3,np.nan,5,6,7,np.nan])
#a[not np.isnan(a)]
a[~ np.isnan(a)]

#如何计算两个数组之间的欧氏距离？

a = np.array([1,2,3,4,5])
b = np.array([4,5,6,7,8])

dist = np.linalg.norm(a-b)


#问题：在一维numpy数组a中查找所有峰值。峰值是两侧较小值包围的点。
a = np.array([1,3,7,1,2,6,0,1])
doublediff = np.diff(np.sign(np.diff(a)))
peak_locations = np.where(doublediff == -2)[0]+1
print peak_locations


#从二维数组a_2d中减去一维数组b_1d，使得每个b_1d项从a_2d的相应行中减去。
a_2d = np.array([[3,3,3],[4,4,4],[5,5,5]])
b_1d = np.array([1,1,1])

print a_2d - b_1d[:, None]



#如何找到数组中第n个重复项的索引, 找出x中第1个重复5次的索引

x=np.array([1,2,1,1,3,4,3,1,1,2,1,1,2])
n=5

np.where(x ==1)[0][n-1]
[i for i, v in enumerate(x) if v == 1][n-1]


#如何将numpy的datetime64对象转换为datetime的datetime对象？
dt64 = np.datetime64('2018-02-25 22:22:22')

from datetime import datetime

dt64.tolist()

dt64.astype(datetime)

#计算给定一维数组窗口大小为3的移动平均值
np.random.seed(100)
Z = np.random.randint(10, size=10)

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] -ret[:-n]
    return ret[n-1:]/n

print moving_average(Z, n=3).round(2)











