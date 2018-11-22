import tensorflow as tf
from os.path import abspath
import re
import tutorial
import numpy as np
from PIL import Image


def model(input):
    """
        This function creates a CNN with two convolution/pooling pairs, followed by two dense layers.
        These are then used to predict one of ten digit classes.

        Arguments:
            input: a tensor with shape `[batch_size, 28, 28, 1]` representing MNIST data.

        Returns:
            A dictionary with the keys `probabilities`, `class`, and `logits`, containing the
            corresponding tensors.
    """

    l = tf.keras.layers
    max_pool = l.MaxPooling2D(
      (2, 2), (2, 2), padding='same', data_format='channels_first')
    # The model consists of a sequential chain of layers, so tf.keras.Sequential
    # (a subclass of tf.keras.Model) makes for a compact description.
    logits = tf.keras.Sequential(
      [
          l.Reshape(
              target_shape=input,
              input_shape=(28 * 28,)),
          l.Conv2D(
              32,
              5,
              padding='same',
              data_format='channels_first',
              activation=tf.nn.relu),
          max_pool,
          l.Conv2D(
              64,
              5,
              padding='same',
              data_format='channels_first',
              activation=tf.nn.relu),
          max_pool,
          l.Flatten(),
          l.Dense(1024, activation=tf.nn.relu),
          l.Dropout(0.4),
          l.Dense(10)
      ])

    # The predictions of our model; we return the logits to formulate a numerically
    # stable loss in our optimization routine
    tensors = {
        "probabilities": tf.nn.softmax(logits),
        "class": tf.argmax(logits, axis=-1),
        "logits": logits
    }
    return tensors

def model2(input):
    """
        This function creates a CNN with two convolution/pooling pairs, followed by two dense layers.
        These are then used to predict one of ten digit classes.

        Arguments:
            input: a tensor with shape `[batch_size, 28, 28, 1]` representing MNIST data.

        Returns:
            A dictionary with the keys `probabilities`, `class`, and `logits`, containing the
            corresponding tensors.
    """

    # we assume the input has shape [batch_size, 28, 28, 1]
    net = input

    # create two convolution/pooling layers
    for i in range(1, 3):
        net = tf.layers.conv2d(inputs=net,
                               filters=32 * i,
                               kernel_size=[5, 5],
                               padding="same",
                               activation=tf.nn.relu,
                               name="conv2d_%d" % i)
        net = tf.layers.max_pooling2d(inputs=net,
                                      pool_size=[2, 2],
                                      strides=2,
                                      name="maxpool_%d" % i)
    # flatten the input to [batch_size, 7 * 7 * 64]
    net = tf.reshape(net, [-1, 7 * 7 * 64])
    net = tf.layers.dense(net, units=128, name="dense_1")
    logits = tf.layers.dense(net, units=10, name="dense_2")
    # logits has shape [batch_size, 10]

    # The predictions of our model; we return the logits to formulate a numerically
    # stable loss in our optimization routine
    tensors = {
        "probabilities": tf.nn.softmax(logits),
        "class": tf.argmax(logits, axis=-1),
        "logits": logits
    }
    return tensors


def training_model(input, label):
    tensors = model(input)

    # one-hot encode the labels, compute cross-entropy loss, average
    # over the number of elements in the mini-batch
    with tf.name_scope("loss"):
        probs = tensors["probabilities"]
        correct_activation = tf.gather(probs, label, axis=-1)
        max_activation = tf.reduce_max(probs, axis=-1)
        non_max_loss = max_activation - correct_activation
        loss = tf.reduce_mean(non_max_loss)

        # label = tf.one_hot(label, 10)
        # loss = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits)
        # loss = tf.reduce_mean(loss)
    tensors["loss"] = loss

    # create the optimizer and register the global step counter with it,
    # it is increased every time the optimization step is executed
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    global_step = tf.train.get_or_create_global_step()
    tensors["opt_step"] = optimizer.minimize(loss, global_step=global_step)
    return tensors


def evaluation_model(input, label):
    tensors = model(input)
    # Let's track some metrics like accuracy, false negatives, and false positives.
    # Each of them returns a tuple `(value, update_op)`. The value is, well, the value
    # of the metric, the update_op is the tensor that needs to be evaluated to update
    # the metric.
    accuracy, update_acc = tf.metrics.accuracy(label, tensors["class"])
    false_negatives, update_fn = tf.metrics.false_negatives(label, tensors["class"])
    false_positives, update_fp = tf.metrics.false_positives(label, tensors["class"])
    tensors["accuracy"] = accuracy
    tensors["false_negatives"] = false_negatives
    tensors["false_positives"] = false_positives
    # We can group the three metric updates into a single operation and run that instead.
    tensors["update_metrics"] = tf.group(update_acc, update_fn, update_fp)
    return tensors


def load_data_set(is_training=True):
    """
        Loads either the training or evaluation dataset by performing the following steps:
        1. Read the file containing the file paths of the images.
        2. Construct the correct labels from the file paths.
        3. Shuffle paths jointly with the labels.
        4. Turn them into a dataset of tuples.

        Arguments:
            is_training: whether to load the training dataset or the evaluation dataset

        Returns:
            The requested dataset as a dataset containing tuples `(relative_path, label)`.
    """
    substring = "training" if is_training else "testing"
    file = "mnist_%s.dataset" % substring

    # this converts the file names into (file_name, class) tuples
    offset = len('mnist_data/%s/c' % substring)
    data_points = []
    with open(file, 'r') as f:
        for line in f:
            cls = int(line[offset])
            path = abspath(line.strip())
            data_points.append((path, cls))

    # now shuffle all data points around and create a dataset from it
    from random import shuffle
    shuffle(data_points)

    # Neat fact: zip can be used to build its own inverse:
    data_points = list(zip(*data_points))

    # note how this is a tensor of strings! Tensors are not inherently numeric!
    images = tf.constant(data_points[0])
    labels = tf.constant(data_points[1])

    # `from_tensor_slices` takes a tuple of tensors slices them along the first
    # dimension. Thus we get a dataset of tuples.
    from tensorflow.python.data.ops import dataset_ops as ops
    return ops.Dataset.from_tensor_slices((images, labels))


def read_image(img_path, label):
    """
        Constructs operations to turn the path to an image to the actual image data.

        Arguments:
            img_path: The path to the image.
            label: The label for the image.

        Returns:
            A tuple `(img, label)`, where `img` is the loaded image.
    """
    img_content = tf.read_file(img_path)
    img = tf.image.decode_png(img_content, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img, label


def make_training_dataset(batch_size=128):
    """
        Creates a training dataset with the given batch size.
    """
    dataset = load_data_set(is_training=True)
    dataset = dataset.repeat(-1)
    dataset = dataset.shuffle(10000)
    dataset = dataset.map(read_image,
                          num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    return dataset


def make_eval_dataset(batch_size=128):
    """
        Creates the evaluation dataset.
    """
    dataset = load_data_set(is_training=False)
    dataset = dataset.map(read_image,
                          num_threads=4,
                          output_buffer_size=100 * batch_size)
    dataset = dataset.batch(batch_size)
    return dataset


def perform_evaluation(checkpoint_name):
    # this is the same as in the training case, except that we are using the
    # evaluation dataset and model
    dataset = make_eval_dataset()
    next_image, next_label = dataset.make_one_shot_iterator().get_next()
    model_outputs = evaluation_model(next_image, next_label)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Initialize variables. This is really necessary here since the metrics
        # need to be initialized to sensible values. They are local variables,
        # meaning that they are not saved by default, which is why we need to run
        # `local_variables_initializer`.
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # restore the weights
        saver.restore(sess, checkpoint_name)

        update_metrics = model_outputs["update_metrics"]
        # feed the inputs until we run out of images
        while True:
            try:
                _ = sess.run(update_metrics)
            except tf.errors.OutOfRangeError:
                # this happens when the iterator runs out of samples
                break

        # Get the final values of the metrics. Note that this call does not access the
        # `get_next` node of the dataset iterator, since these tensors can be evaluated
        # on their own and therefore don't cause the exception seen above to be triggered
        # again.
        accuracy = model_outputs["accuracy"]
        false_negatives = model_outputs["false_negatives"]
        false_positives = model_outputs["false_positives"]
        acc, fp, fn = sess.run((accuracy, false_negatives, false_positives))
        print("Accuracy: %f" % acc)
        print("False positives: %d" % fp)
        print("False negatives: %d" % fn)

'''
def perform_training(steps, batch_size, checkpoint_name, logdir):
    dataset = make_training_dataset(batch_size)
    next_image, next_label = dataset.make_one_shot_iterator().get_next()
    model_outputs = training_model(next_image, next_label)

    loss = model_outputs["loss"]
    opt_step = model_outputs["opt_step"]

    # generate summary tensors, but don't store them -- there's a better way
    frequent_summary = tf.summary.scalar("loss", loss)
    tf.summary.histogram("logits", model_outputs["logits"])
    tf.summary.histogram("probabilities", model_outputs["probabilities"])
    # only log one of the `batch_size` many images
    tf.summary.image("input_image", next_image, max_outputs=1)

    # merge all summary ops into a single operation
    infrequent_summary = tf.summary.merge_all()

    variables_to_save = get_variables_for_saver(exclude=[".*Adam", ".*beta", ".*Iterator"])
    saver = tf.train.Saver(variables_to_save)

    # the summary writer will write the summaries to the specified log directory
    summary_writer = tf.summary.FileWriter(logdir=logdir, graph=tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(steps):
            # note that we explicitly ask for the evaluation of the summary ops
            if i % 10 == 0:
                _, summary = sess.run((opt_step, infrequent_summary))
            else:
                _, summary = sess.run((opt_step, frequent_summary))
            # ...and we also need to explicitly add them to the summary writer
            summary_writer.add_summary(summary, global_step=i)

        saver.save(sess, checkpoint_name)
        summary_writer.close()
'''

# This function filters the model's variables by some regular expression
def get_variables_for_saver(include=None, exclude=None):
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    if include is []:
        print("The include list for the initilization is the empty list. "
              "Is that intentional? No variables are saved/loaded!")
    elif include is not None:
        regexs = map(re.compile, include)
        variables = filter(lambda v: any(map(lambda r: r.match(v.name), regexs)), variables)
    if exclude is not None:

        for e in exclude:
            regex = re.compile(e)
            variables = filter(lambda v: not regex.match(v.name), variables)
    return list(variables)

'''
tutorial.perform_training_with_visualization(steps=500, batch_size=128,
                                             dataset=make_training_dataset,
                                             model=training_model)

'''
def perform_training(steps, batch_size, checkpoint_name):
    dataset = make_training_dataset(batch_size)
    next_image, next_label = dataset.make_one_shot_iterator().get_next()   
    model_outputs = training_model(next_image, next_label)
    
    loss = model_outputs["loss"]
    opt_step = model_outputs["opt_step"]
    
    # We exclude some variables created by the optimizer and data loading process
    # when saving the model, since we assume that we do not want to continue training
    # from the checkpoint.
    variables_to_save = tutorial.get_variables_for_saver(exclude=[".*Adam", ".*beta", ".*Iterator"])
    saver = tf.train.Saver(variables_to_save)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(steps):
            _, loss_value = sess.run((opt_step, loss))
        
        # In the end, we  will save the model
        saver.save(sess, "./"+checkpoint_name)
        print("Final Loss: %f" % loss_value)


tf.reset_default_graph()
perform_training(steps=500, batch_size=128,
                 checkpoint_name="model")


def image_classifier(checkpoint_name, *images):
    # note that we are deciding to feed the images one-by-one for simplicity
    input = tf.placeholder(name="input", shape=(1, 28, 28, 1), dtype="float32")
    tensors = model(input)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Initialize all variables, since there may be variables in our graph that aren't
        # loaded form the checkpoint
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_name)
        
        classes = []
        for i in images:
            cls = sess.run(tensors["class"], feed_dict={input: np.expand_dims(i, 0)})
            classes.append(cls[0])
        return classes

def load_image_as_array(filepath):
    im = Image.open(abspath(filepath)).convert('L')
    (width, height) = im.size
    greyscale_map = list(im.getdata())
    greyscale_map = np.array(greyscale_map)
    greyscale_map = greyscale_map.reshape((height, width, 1))
    return greyscale_map


tf.reset_default_graph()
images = [
    "mnist_data/testing/c1/994.png",
    "mnist_data/testing/c3/216.png",
    "mnist_data/testing/c4/931.png"
]
classification = image_classifier("model", *map(load_image_as_array, images))
print("Classification: " + str(classification))


tf.reset_default_graph()
perform_evaluation(checkpoint_name="model")