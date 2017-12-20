# coding: UTF-8

import tensorflow as tf

# 加载mnist数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 初始化session
sess = tf.InteractiveSession()

def weight_variable(shape):
    #权重为标准差为0.1的正态分布随机样本
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # 偏置设置0.1为了防止ReLU的死亡节点
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool_2x2(x):
    return  tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

#输入的训练集
x=tf.placeholder(tf.float32,[None,784])
#训练集标签
y_=tf.placeholder(tf.float32,[None,10])
#mnist的数据集是1维的,因此要把其转换成4维 1x784->28x28
x_image=tf.reshape(x,[-1,28,28,1])

#[5,5,1,32]表示使用卷积核尺寸为5x5,1个颜色通道，32个不同的卷积核
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)


#较第一层，唯一不同的就是卷积核的数目变成了64
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
# (n, 7, 7, 64) ->(n,7*7*64) n为样本数
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#为了减轻过过拟合,在训练时，根据输入的比率随机丢弃一部分节点
kepp_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob=kepp_prob)

W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化所有参数
tf.global_variables_initializer().run()
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 进行20000次迭代训练
for i in range(2000):
    # 随机从训练集中抽取50个样本作为一个批次进行训练
    batch_xs, batch_ys = mnist.train.next_batch(50)
    if i%1000==0:
        train_accuracy=accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, kepp_prob: 1.0})
        print("step %d,training accurary %g"%(i,train_accuracy))
    # 为占位变量赋值
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, kepp_prob: 0.5})

# 计算模型的准确度
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, kepp_prob: 1.0}))

