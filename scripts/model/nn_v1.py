# Code sourced from Coursera deep learning specialization exercise
import pickle
import tensorflow as tf
import seaborn as sns
import matplotlib as plt

# GPU is available
print(tf.test.is_gpu_available())
print(tf.test.is_built_with_cuda())

###############################################################################
# Set up folders and variables
###############################################################################
filename = 'nn_v1.py'

save_folder = '../../data/data_for_model/'
load_folder = '../../data/data_for_model/'
model_folder = '../../data/model/'
figure_folder = '../../images/model/nn1/'

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
# Build Keras model
###############################################################################


model = tf.keras.models.Sequential([
  tf.keras.Input(shape=(x_train_sel_df.shape[1],)),
  tf.keras.layers.Dense(6, activation='relu'),
  tf.keras.layers.Dense(4, activation='relu'),
  tf.keras.layers.Dense(3, activation='softmax'),
  ])

# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
# Instantiate a loss function.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

###############################################################################
# Minibatch setup
###############################################################################

# Prepare the training dataset.
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_sel_df.values,
                                                    y_train_sel_df.values))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the test dataset.
test_dataset = tf.data.Dataset.from_tensor_slices(
  (x_test_sel_df.values, y_test_sel_df.values))
test_dataset = test_dataset.batch(batch_size)

###############################################################################
# Train
###############################################################################

train_loss_results = []
train_accuracy_results = []
test_accuracy_results = []

# Iterate over epochs.
epochs = 1000
for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))
    epoch_loss_avg = tf.keras.metrics.Mean()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Update training metric.
        train_acc_metric(y_batch_train, logits)
        # Track progress
        epoch_loss_avg(loss_value)  # Add current batch loss

        # Log every 200 batches.
        if step % 200 == 0:
            print('Training loss (for one batch) at step %s: %s'
                  % (step, float(loss_value)))
            print('Seen so far: %s samples' % ((step + 1) * batch_size))

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print('Training acc over epoch: %s' % (float(train_acc),))
    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Run a test loop at the end of each epoch.
    for x_batch_test, y_batch_test in test_dataset:
        test_logits = model(x_batch_test)
        # Update val metrics
        test_acc_metric(y_batch_test, test_logits)
    test_acc = test_acc_metric.result()
    test_acc_metric.reset_states()
    print('Test acc: %s' % (float(test_acc),))

    # End epoch - append loss and accuracy
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(train_acc)
    test_accuracy_results.append(test_acc)


model.save(model_folder+'nn1/')
model = tf.keras.models.load_model(model_folder+'nn1/')
model.summary()

# =============================================================================
# Training acc over epoch: 0.9819875955581665
# Test acc: 0.9826302528381348
# =============================================================================

###############################################################################
# Plot the loss and accuracy function
###############################################################################

train_loss_results_list = [float(x) for x in train_loss_results]
train_accuracy_results_list = [float(x) for x in train_accuracy_results]
test_accuracy_results_list = [float(x) for x in test_accuracy_results]

fig, axes = plt.pyplot.subplots(3, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].set_xlabel("Epoch", fontsize=14)
axes[0].plot(train_loss_results_list)

axes[1].set_ylabel("Train Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results_list)

axes[2].set_ylabel("Test Accuracy", fontsize=14)
axes[2].set_xlabel("Epoch", fontsize=14)
axes[2].plot(test_accuracy_results_list)

plt.pyplot.show()

fig.savefig(figure_folder+'training_metrics.png')
fig.clf()

###############################################################################
# Heatmap of the weights
###############################################################################

model.get_weights()[0]  # W1
model.get_weights()[1]  # b1
model.get_weights()[2]  # W2
model.get_weights()[3]  # b2
model.get_weights()[4]  # W3
model.get_weights()[5]  # b3

ax = sns.heatmap(model.get_weights()[0])
fig = ax.get_figure()
fig.savefig(figure_folder+'weights_layer1.png')
fig.clf()
ax = sns.heatmap(model.get_weights()[2])
fig = ax.get_figure()
fig.savefig(figure_folder+'weights_layer2.png')
fig.clf()
ax = sns.heatmap(model.get_weights()[4])
fig = ax.get_figure()
fig.savefig(figure_folder+'weights_layer3.png')
fig.clf()
