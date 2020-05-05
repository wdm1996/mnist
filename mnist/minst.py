from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
def mnist_demo():
    # 加载数据集
    mnist = input_data.read_data_sets('D:/MNIST_data', one_hot=True)
    # images, labels = mnist.train.next_batch(100)
    # 1 准备数据
    x = tf.placeholder(tf.float32, shape=[None, 784])  # ------------------使用placeholder构建变量
    y_true = tf.placeholder(tf.float32, shape=[None, 10])
    # 2 构建模型----------------------------------------x(None,784)*weight(784,10)+bias=y(None,10)
    weight = tf.Variable(tf.zeros(shape=[784, 10]))
    bias = tf.Variable(initial_value=tf.zeros(shape=[10]))
    y_predict = tf.nn.softmax(tf.matmul(x, weight)+bias)
    # 3 构建损失函数-----------------------------------------多分类问题，使用交叉熵损失函数 softmax()
    cross_entropy = -tf.reduce_sum(y_true*tf.log(y_predict))
    # 4 优化损失------------------------------------------------------使用梯度下降
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    # 初始化变量
    # init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    # 开启会话
    with tf.Session() as sess:
        # 运行初始化变量
        sess.run(init)
        for j in range(10):  # 重复10次
            aver = 0.
        # 训练1000次
            for i in range(1000):
                images, labels = mnist.train.next_batch(100)
                sess.run(train_step, feed_dict={x: images, y_true: labels})
            # print('第%d次训练模型的损失：%f' % ((i+1), loss))

            correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_true, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("正确率：%.3f" % (100*sess.run(accuracy, feed_dict={x: mnist.test.images, y_true: mnist.test.labels})), "%")
            aver = sess.run(accuracy, feed_dict={x: mnist.test.images, y_true: mnist.test.labels})
        print("平均正确率：%.3f" % (aver*100), "%")

if __name__ == '__main__':
    mnist_demo()