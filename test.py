import  torch
import  tensorflow as tf
from    torch.nn import functional as F


tf.enable_eager_execution()








def main():

    y = torch.rand((3,20))
    x = torch.randint(0, 2, (3,20))

    print(x)
    print(y)

    loss = torch.sum(x * torch.log(y) + (1 - x) * torch.log(1 - y)) / 3
    loss2 = -F.binary_cross_entropy(y, x, reduction='sum') / 3


    x = tf.Variable(x.numpy())
    y = tf.Variable(y.numpy())

    loss3 = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y)) / 3
    y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)
    loss4 = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y)) / 3


    print(loss, loss2, loss3, loss4)


if __name__ == '__main__':
    main()


