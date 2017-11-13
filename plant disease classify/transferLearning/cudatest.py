import tensorflow as tf
x=tf.constant(5)
y=tf.constant(35)

import torch
x=torch.cuda
y=tf.constant(35)
a=torch.Tensor(5,4)
print(a)
b=a.cuda()
print(b)

