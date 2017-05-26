import tensorflow as tf
import numpy as np
import ray

ray.init()

BATCH_SIZE = 100
NUM_BATCHES = 1
NUM_ITERS = 201

class Network(object):
  def __init__(self, x, y):
    # Seed TensorFlow to make the script deterministic.
    tf.set_random_seed(0)
    # Define the inputs.
    x_data = tf.constant(x, dtype=tf.float32)
    y_data = tf.constant(y, dtype=tf.float32)
    # Define the weights and computation.
    w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    y = w * x_data + b
    # Define the loss.
    self.loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    self.grads = optimizer.compute_gradients(self.loss)
    self.train = optimizer.apply_gradients(self.grads)
    # Define the weight initializer and session.
    init = tf.global_variables_initializer()
    self.sess = tf.Session()
    # Additional code for setting and getting the weights
    self.variables = ray.experimental.TensorFlowVariables(self.loss, self.sess)
    # Return all of the data needed to use the network.
    self.sess.run(init)

  # Define a remote function that trains the network for one step and returns the
  # new weights.
  def step(self, weights):
    # Set the weights in the network.
    self.variables.set_weights(weights)
    # Do one step of training. We only need the actual gradients so we filter over the list.
    actual_grads = self.sess.run([grad[0] for grad in self.grads])
    return actual_grads

  def get_weights(self):
    return self.variables.get_weights()

# Define a remote function for generating fake data.
@ray.remote(num_return_vals=2)
def generate_fake_x_y_data(num_data, seed=0):
  # Seed numpy to make the script deterministic.
  np.random.seed(seed)
  x = np.random.rand(num_data)
  y = x * 0.1 + 0.3
  return x, y

# Generate some training data.
batch_ids = [generate_fake_x_y_data.remote(BATCH_SIZE, seed=i) for i in range(NUM_BATCHES)]
x_ids = [x_id for x_id, y_id in batch_ids]
y_ids = [y_id for x_id, y_id in batch_ids]
# Generate some test data.
x_test, y_test = ray.get(generate_fake_x_y_data.remote(BATCH_SIZE, seed=NUM_BATCHES))

# Create actors to store the networks.
remote_network = ray.remote(Network)
actor_list = [remote_network.remote(x_ids[i], y_ids[i]) for i in range(NUM_BATCHES)]
local_network = Network(x_test, y_test)

# Get initial weights of local network.
weights = local_network.get_weights()

# Do some steps of training.
for iteration in range(NUM_ITERS):
  # Put the weights in the object store. This is optional. We could instead pass
  # the variable weights directly into step.remote, in which case it would be
  # placed in the object store under the hood. However, in that case multiple
  # copies of the weights would be put in the object store, so this approach is
  # more efficient.
  weights_id = ray.put(weights)
  # Call the remote function multiple times in parallel.
  gradients_ids = [actor.step.remote(weights_id) for actor in actor_list]
  # Get all of the weights.
  gradients_list = ray.get(gradients_ids)

  # Take the mean of the different gradients. Each element of gradients_list is a list
  # of gradients, and we want to take the mean of each one.
  mean_grads = [sum([gradients[i] for gradients in gradients_list]) / len(gradients_list) for i in range(len(gradients_list[0]))]

  feed_dict = {grad[0]: mean_grad for (grad, mean_grad) in zip(local_network.grads, mean_grads)}
  local_network.sess.run(local_network.train, feed_dict=feed_dict)
  weights = local_network.get_weights()

  # Print the current weights. They should converge to roughly to the values 0.1
  # and 0.3 used in generate_fake_x_y_data.
  if iteration % 20 == 0:
    print("Iteration {}: weights are {}".format(iteration, weights))