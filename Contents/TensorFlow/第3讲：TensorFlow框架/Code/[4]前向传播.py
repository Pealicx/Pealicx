# coding:utf-8
import tensorflow as tf

# 两层神经网络的全连接

# 定义输入和参数
# 1 x = tf.constant([[0.7, 0.5]])
# 2 x = tf.placeholder(tf.float32, shape=[1, 2])
x = tf.placeholder(tf.float32, shape=[None, 2])
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义计算图
a = tf.matmul(x, w1)
b = tf.matmul(a, w2)

# 会话
with tf.Session() as sess:
    # 初始化变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 1 ans = sess.run(y)
    # 2 ans = sess.run(b, feed_dict={x: [[0.7, 0.5]]})
    ans = sess.run(b, feed_dict={x: [[0.7, 0.5], [0.8, 0.6], [0.7, 0.9]]})
    print(ans)
