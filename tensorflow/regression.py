import tensorflow as tf
import numpy as np

## prepare the original data
x_data = np.random.rand(100).astype(np.float32)
y_data = 0.3*x_data+0.1
##creat parameters
weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))
bias = tf.Variable(tf.zeros([1]))
##get y_prediction
y_prediction = weight*x_data+bias
##compute the loss
loss = tf.reduce_mean(tf.square(y_data-y_prediction))
##creat optimizer
optimizer = tf.train.GradientDescentOptimizer(0.5)
#creat train ,minimize the loss 
train = optimizer.minimize(loss)
#creat init 
init = tf.global_variables_initializer()

##creat a Session 
sess = tf.Session()
##initialize
sess.run(init)
## Loop
for step in range(101):
    sess.run(train)
    if step %10==0 :
        print (step,'weight:',sess.run(weight),'bias:',sess.run(bias))