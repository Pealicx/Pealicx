# coding:utf:8
# 酸奶的销量Y_和属性x1,x2的关系是Y_= x1 + x2,随机噪声是0.05
import tensorflow as tf
import numpy as np

BATCH_SIZE = 8  # 每一次喂入神经网络的数据的多少
SEED = 23455
COST = 9
PROFIT = 1

# 认为定义原始数据
rdm = np.random.RandomState(SEED)  # 定义随机函数
X = rdm.rand(32, 2)  # rand函数每次生成[0,1)的随机数
Y_ = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in X]  # 标准销量

# 定义神经网络的输入、参数、输出、计算图
# 该神经网络一共三个神经元，两个输入，一个输出
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))  # 随机参数列表
y = tf.matmul(x, w1)  # 预测出来的答案

# 定义损失函数为：MSE(均方误差)
loss_mse1 = tf.reduce_mean(tf.square(y - y_))
# 自定义损失函数
loss_mse2 = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * COST, (y_ - y) * PROFIT))

# 定义反向传播为：梯度下降
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss_mse2)

# 生成会话，训练STEEP论述
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 20000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 500 == 0:
            print("After %d training steps, w1 is\n" % i, sess.run(w1))
    print(sess.run(w1))
