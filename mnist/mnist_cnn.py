from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets('D:/MNIST_data', one_hot=True) #加载数据集
'''
data(n*784) -> 使用64个filter(3*3*1) + conv1 -> max(2*2)  pooling -> 使用128个fliter(3*3*64)+conv2
-> max(2*2) pooling ->经过poling变成了7*7*128，全连接层 fc1(7*7*128,1024) -> 全连接层fc2(1024,10)
'''
#print(help(tf.cast)) #------------------------------------------使用print help 函数在下面命令窗口查看tf.cast帮助文档

def weight_variable(shape):                         #这个模型中的权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度
  initial = tf.truncated_normal(shape, stddev=0.1)  #使用tf.truncated_normal(shape,mean,stddev) 截断的产生正态分布的随机数，即随机数与均值的差若大于两倍的标准差则重新生成
  return tf.Variable(initial)                       #shape：生成张量的维度 ，mean：均值 ，stddev：标准差

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)



def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # 使用步长为1，边距为0的模板

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') # 使用max_pooling ，2x2大小



x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
#------------------------------------------------------------------------------第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32])  # 算出32个特征图
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 使用relu激活函数
h_pool1 = max_pool_2x2(h_conv1)  # max_pooling
#------------------------------------------------------------------------------ 第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])  # 算出64个特征图
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)   # 使用relu激活函数
h_pool2 = max_pool_2x2(h_conv2)
#-------------------------------------------------------------------------------全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])  #使用1024个神经元
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])  #将池化层输出的张量reshape成一些向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  #使用relu函数输出
#-------------------------------------------------------------------------------使用dropout防止过拟合
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#---------------------------------------------------------------------------------添加softmax层输出
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#----------------------------------------------------ADAM优化器来做梯度最速下降，在feed_dict中加入额外的参数keep_prob来控制dropout比例
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))  # 损失函数
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 优化损失
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        if(i%200 == 0):
            print("step %d, training accuracy %.2f" % (i, 100*sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})) + "%")
    print("test accuracy %.2f" % (100*sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})) + "%")
