'''
指定两层 L1(256个神经元)和L2(128个神经元)
w1 = 784*256 b1 = 256
w2 = 256*128 b2 = 128
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets('D:/MNIST_data', one_hot=True) #加载数据集

n_hidden_1 = 256  # L1(256个神经元)
n_hidden_2 = 128  # L2(128个神经元)
n_input = 784
n_classes = 10

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

stddev = 0.1  #指定方差项
weights = {  #使用高斯分布初始化
    'w1': tf.Variable(tf.random_normal([784, 256], stddev=stddev)),
    'w2': tf.Variable(tf.random_normal([256, 128], stddev=stddev)),
    'out': tf.Variable(tf.random_normal([128, 10], stddev=stddev))
}
biases = {
    'b1': tf.Variable(tf.random_normal([256])),
    'b2': tf.Variable(tf.random_normal([128])),
    'out': tf.Variable(tf.random_normal([10]))
}

layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['w1']), biases['b1']))
layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['w2']), biases['b2']))
pred = (tf.matmul(layer_2, weights['out']) + biases['out'])
#预测值pred

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optm = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(corr, "float"))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for j in range(10):
        for i in range(10000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
          # if ((i % 1000)==0):
               # print("第%d次训练" % i)
               #print("trian accuracy: %.3f" % (sess.run(accr, feed_dict={x: batch_xs, y: batch_ys})))
        print("test accuracy: %.2f" % (100*sess.run(accr, feed_dict={x: mnist.test.images, y: mnist.test.labels})), '%')