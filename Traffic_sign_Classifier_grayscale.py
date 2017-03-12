
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
#
# ## Deep Learning
#
# ## Project: Build a Traffic Sign Recognition Classifier
#
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary.
#
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
#
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
#
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
#
#
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[ ]:

# Load pickled data
import pickle
import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import cv2

# TODO: Fill this in based on where you saved the training and testing data

training_file = "./traffic-sign-data/train.p"
validation_file= "./traffic-sign-data/valid.p"
testing_file =  "./traffic-sign-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))


print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.

# Visualizations will be shown in the notebook.
#%matplotlib inline
n, bins, patches = plt.hist(y_train, n_classes, normed=0)
plt.show()

#get single occurances and first occurance to temp_indices of each class in a vector
temp, temp_indices = np.unique(y_train,return_index=True)
#for i in range(len(temp)):
    #class_number = temp[i]
    #class_first_occurance = temp_indices[i]
    #number_of_elements = y_train.count(i)
    #index = random.randint(0, len(X_train))
    #image = X_train[class_first_occurance].squeeze()
    #plt.figure(figsize=(5,5))
    #plt.imshow(image, cmap="gray")
    #plt.title(y_train[class_first_occurance])
    #print(y_train[class_first_occurance])

#preprocessing the data
#step1) Convert to grayscale
#step2) Zero mean
X_train_grayscale = np.zeros((X_train.shape[0],X_train.shape[1],X_train.shape[2]))
X_valid_grayscale = np.zeros((X_valid.shape[0],X_valid.shape[1],X_valid.shape[2]))
for i in range(X_train.shape[0]):
    X_train_grayscale[i] = cv2.cvtColor(X_train[i], cv2.COLOR_RGB2GRAY)
    X_train_grayscale[i] = (X_train_grayscale[i] - np.mean(X_train_grayscale[i])) / (
    np.max(X_train_grayscale[i]) - np.min(X_train_grayscale[i]))

for i in range(X_valid.shape[0]):
    X_valid_grayscale[i] = cv2.cvtColor(X_train[i], cv2.COLOR_RGB2GRAY)
    X_valid_grayscale[i] = (X_valid_grayscale[i] - np.mean(X_valid_grayscale[i])) / (
    np.max(X_valid_grayscale[i]) - np.min(X_valid_grayscale[i]))

from tensorflow.contrib.layers import flatten

def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    weights_c1 = tf.Variable(tf.truncated_normal([5,5,3,6],mu,sigma))
    bias_c1 = tf.Variable(tf.zeros(6))
    convolution_stride = np.array([1,1,1,1])
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    x = tf.nn.conv2d(x,weights_c1,[1,1,1,1],'VALID')
    x = tf.nn.bias_add(x,bias_c1)


    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    x = tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],'VALID')

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    weights_c2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16], mu, sigma))
    bias_c2 = tf.Variable(tf.zeros(16))
    convolution_stride2 = np.array([1, 1, 1, 1])

    x = tf.nn.conv2d(x, weights_c2, [1,1,1,1], 'VALID')
    x = tf.nn.bias_add(x, bias_c2)

    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1],'VALID')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    x = flatten(x)

    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    weights1 = tf.Variable(tf.truncated_normal([400,120],mu,sigma))
    bias1 = tf.Variable(tf.zeros(120))
    x = tf.add(tf.matmul(x,weights1),bias1)

    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    weights2 = tf.Variable(tf.truncated_normal([120, 84], mu, sigma))
    bias2 = tf.Variable(tf.zeros(84))
    x = tf.add(tf.matmul(x, weights2), bias2)

    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    weights3 = tf.Variable(tf.truncated_normal([84, 43], mu, sigma))
    bias3 = tf.Variable(tf.zeros(43))
    logits = tf.add(tf.matmul(x, weights3), bias3)
    return logits

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


EPOCHS = 10
BATCH_SIZE = 128

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train_grayscale, y_train = shuffle(X_train_grayscale, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")
