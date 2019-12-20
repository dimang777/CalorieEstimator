# Code sourced from Coursera deep learning specialization exercise
import pickle
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.framework import ops
import seaborn as sns
import matplotlib as plt
import numpy as np

# GPU is available
print(tf.test.is_gpu_available())
print(tf.test.is_built_with_cuda())

###############################################################################
# Set up folders and variables
###############################################################################
filename = 'neural_network.py'

save_folder = '../../data/data_for_model/'
load_folder = '../../data/data_for_model/'
model_folder = '../../data/model/'
figure_folder = '../../images/model/nn/'

###############################################################################
# Load
###############################################################################

with open(save_folder + 'train_test_sel_features.pkl', 'rb') as f:
    [features,
        excluded_features,
        x_train_sel_df,
        x_test_sel_df,
        y_train_sel_df,
        y_test_sel_df,
        train_idx,
        test_idx,
        train_flag,
        test_flag] = pickle.load(f)

###############################################################################
# Functions
###############################################################################


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat),
         of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- consistency

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    # number of training examples
    m = X.shape[1]
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = np.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[
          :, k * mini_batch_size:k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[
          :, k * mini_batch_size:k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[
          :, num_complete_minibatches * mini_batch_size:m]
        mini_batch_Y = shuffled_Y[
          :, num_complete_minibatches * mini_batch_size:m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- scalar, size of an input vector
    n_y -- scalar, number of classes (from 0 to 2, so -> 3)

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels,
         of shape [n_y, None] and dtype "float"

    Tips:
    - None used because of the flexibility on the number of examples.
      I.e., The number of examples during test/train is different.
    """

    X = tf.placeholder(tf.float32, shape=[n_x, None],
                       name='Placeholder_1')
    Y = tf.placeholder(tf.float32, shape=[n_y, None],
                       name='Placeholder_2')

    return X, Y


def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow.
    The shapes are:
                        W1 : [6, 11]
                        b1 : [6, 1]
                        W2 : [4, 6]
                        b2 : [4, 1]
                        W3 : [3, 4]
                        b3 : [3, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    tf.set_random_seed(1)

    W1 = tf.get_variable(
      'W1', [6, 11],
      initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable(
      'b1', [6, 1],
      initializer=tf.zeros_initializer())
    W2 = tf.get_variable(
      'W2', [4, 6],
      initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable(
      'b2', [4, 1],
      initializer=tf.zeros_initializer())
    # Should match # of class
    W3 = tf.get_variable(
      'W3', [3, 4],
      initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable(
      'b3', [3, 1],
      initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters
                  "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.matmul(W1, X) + b1   # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)          # A1 = relu(Z1)
    Z2 = tf.matmul(W2, A1) + b2  # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)          # A2 = relu(Z2)
    Z3 = tf.matmul(W3, A2) + b3  # Z3 = np.dot(W3,Z2) + b3

    return Z3


def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit),
    of shape (3, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    # to fit the tensorflow requirement for
    # tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits
      (logits=logits, labels=labels))

    return cost


def model(x_train, y_train, x_test, y_test, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=32, print_cost=True):
    """
    Implements a three-layer tensorflow neural network:
    LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    x_train -- train set, of shape
              (input size = 11, number of training examples = 1610)
    y_train -- train set, of shape
              (output size = 3, number of training examples = 1610)
    x_test -- test set, of shape
              (input size = 11, number of training examples = 403)
    y_test -- test set, of shape
              (output size = 3, number of test examples = 403)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learned by the model.
                  They can then be used to predict.
    """

    # to be able to rerun the model without overwriting tf variables
    ops.reset_default_graph()
    # to keep consistent results
    tf.set_random_seed(1)
    seed = 3
    # (n_x: input size, m : number of examples in the train set)
    (n_x, m) = x_train.shape
    # n_y : output size
    n_y = y_train.shape[0]
    # To keep track of the cost
    costs = []

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters()

    # Forward propagation:
    # Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = \
        tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.  # Defines a cost related to an epoch
            # number of minibatches of size minibatch_size in the train set
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = \
                random_mini_batches(x_train, y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost",
                # the feed_dict should contain a minibatch for (X,Y).
                _, minibatch_cost = sess.run(
                      [optimizer, cost],
                      feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per five)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        plt.savefig(figure_folder+'Cost_nn.png')  # didn't work

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: x_train, Y: y_train}))
        print("Test Accuracy:", accuracy.eval({X: x_test, Y: y_test}))

        return parameters

###############################################################################
# Assign variables
###############################################################################


# Transpose the matrix for tf nn
x_train = x_train_sel_df.values.T
x_test = x_test_sel_df.values.T

###############################################################################
# One hot encoding of the labels
###############################################################################


# Make the array explicitly (n,1) size
y_train_ohe_prep = y_train_sel_df.to_numpy().reshape(len(y_train_sel_df), 1)
y_test_ohe_prep = y_test_sel_df.to_numpy().reshape(len(y_test_sel_df), 1)
ohe = OneHotEncoder(sparse=False)
# Transpose the matrix for tf nn
y_train = ohe.fit_transform(y_train_ohe_prep).T
y_test = ohe.fit_transform(y_test_ohe_prep).T

print("number of training examples = " + str(x_train.shape[1]))
print("number of test examples = " + str(x_test.shape[1]))
print("x_train shape: " + str(x_train.shape))
print("y_train shape: " + str(y_train.shape))
print("x_test shape: " + str(x_test.shape))
print("y_test shape: " + str(y_test.shape))

###############################################################################
# Set up and train NN architecture
###############################################################################

parameters = model(x_train, y_train, x_test, y_test)

type(parameters)
W1 = parameters['W1']
W2 = parameters['W2']
W3 = parameters['W3']

ax = sns.heatmap(W1)
ax = sns.heatmap(W2)
ax = sns.heatmap(W3)

with open(model_folder+'nn_parameters.pkl', 'wb') as f:
    pickle.dump([parameters], f)

with open(model_folder+'nn_parameters.pkl', 'rb') as f:
    parameters = pickle.load(f)
