# coding:utf-8
# 导入np模块，生成模拟数据集
import tensorflow as tf
import numpy as np

# 每次喂入神经网络的数据的组数，不宜过大
BATCH_SIZE = 8
SEED = 23455

# 基于SEED产生随机数
rng = np.random.RandomState(SEED)

# 随机数产生32行2列的随机数，表示3组数据，每组两个参数：体积和重量，作为样本特征,[0,1)之间
X = rng.rand(32, 2)

# 人工标记这32组特征，若体积+中重量 < 1 ，标记为1，否者标记为0

Y = [[int(x0 + x1 < 0)] for (x0, x1) in X]

# 定义输入输出
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2, 3], stddev=2, mean=0, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=2, mean=0, seed=1))

# 搭建计算图,注意参数的传递和意义
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数以及参数优化的方法
loss = tf.reduce_mean(tf.square(y - y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# train_step = tf.train.MomentumOptimizer(0.001).minimize(loss)
# train_step = tf.train.AdadeltaOptimizer(0.001).minimize(loss)

# 生成会话，训练STEPS次
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 输出训练前NN参数
    print("W1\n", sess.run(w1))
    print("W2\n", sess.run(w2))
    print("\n")

    # 训练模型
    STEPS = 3000
    for i in range(STEPS):
        st = (i * BATCH_SIZE) % 32
        ed = st + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[st:ed], y_: Y[st:ed]})
        if (i % 500) == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print("After %d training steps,total loss on al data is %g", (i, total_loss))

    # 输出训练后NN参数
    print("W1\n", sess.run(w1))
    print("W2\n", sess.run(w2))
    print("\n")
