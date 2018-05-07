#coding:utf-8
import tensorflow as TF 
#定义张量
a = TF.constant([[2.0,2.0]])
b = TF.constant([[2.0],[2.0]])
#定义计算图
ans = TF.matmul(a,b)

print ans
#计算结果 ：Tensor("MatMul:0", shape=(1, 1), dtype=float32)

#会话
with TF.Session() as sess:
	print sess.run(ans)
#计算结果：[[8.]]