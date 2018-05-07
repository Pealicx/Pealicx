import tensorflow as TF 
a = TF.constant([1.0,2.0])
b = TF.constant([3.0,4.0])
ans = a + b
print ans

# Tensor("add:0", shape=(2,), dtype=float32)
# 		 节点名	  维度		  数据类型