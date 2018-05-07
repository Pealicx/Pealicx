# coding:utf-8
import tensorflow as tf

# 随机数生成：正太分布生成2*3的矩阵，标准差是2，均值为0，随机种子是1。后三个参数可以不写
w1 = tf.Variable(tf.random_normal([2, 3], stddev=2, mean=0, seed=1))

# 随机数生成：去掉过大偏离点的正太分布生成2*3的矩阵，标准差是2，均值为0，随机种子是1。后三个参数可以不写
w2 = tf.Variable(tf.truncated_normal([2, 3], stddev=2, mean=0, seed=1))

# 随机数生成：平均分布生成2*3的矩阵。
w3 = tf.Variable(tf.random_uniform([2, 3]))

# 生成全零参数
w4 = tf.zeros([3, 3], tf.int32)

# 生成全一参数
w5 = tf.ones([3, 3], tf.int32)

# 生成全固定参数
w6 = tf.fill([3, 3], 6)

# 参数直接赋值
w7 = tf.constant([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
# 会话：
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("w1")
    print(sess.run(w1))
    print("w2")
    print(sess.run(w2))
    print("w3")
    print(sess.run(w3))
    print("w4")
    print(sess.run(w4))
    print("w5")
    print(sess.run(w5))
    print("w6")
    print(sess.run(w6))
    print("w7")
    print(sess.run(w7))
