import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-1,1,300) [:, np.newaxis]   # [:,np.newaxis] make row vector transform column vector
noise = np.random.normal(0, 0.05, x_data.shape) 
y_data = np.square(x_data) - 0.5 + noise

fig=plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.xlabel('x_data')
plt.ylabel('y_data')
plt.show()

def add_nn_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])


# add hidden layer
l1 = add_nn_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_nn_layer(l1, 10, 1, activation_function=None)


#compute loss
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
# creat train op,then we can sess.run(train)  to minimize loss
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# creat init 
init = tf.global_variables_initializer()

# creat a Session
sess = tf.Session()

# system initialize
sess.run(init)

# training 
for i in range(1000):
    sess.run(train, feed_dict={xs: x_data, ys: y_data})
    prediction_value = sess.run(prediction, feed_dict={xs: x_data})
    if i % 50 == 0:
        # to see the step improvement
        print('loss:',sess.run(loss, feed_dict={xs: x_data, ys: y_data}))


fig=plt.figure()
bx = fig.add_subplot(1,1,1)
bx.scatter(x_data,y_data)
bx.plot(x_data,prediction_value,'g-',lw=6)
plt.xlabel('x_data')
plt.ylabel('y_data')
plt.show()