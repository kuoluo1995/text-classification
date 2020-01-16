import numpy as np
import tensorflow as tf


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def gcn_layer(x, support, in_channels, out_channels, keep_pro, is_sparse, num_nonzero=None, name='gcn_layer'):
    vars = {}
    with tf.variable_scope('{}_vars'.format(name)):
        for i in range(len(support)):
            vars['weights_{}'.format(i)] = glorot([in_channels, out_channels], name='weights_{}'.format(i))
        if is_sparse:
            x = sparse_dropout(x, keep_pro, num_nonzero)
        else:
            x = tf.nn.dropout(x, keep_pro)
        # convolve
        supports = list()
        for i in range(len(support)):
            if is_sparse:
                pre_sup = vars['weights_{}'.format(i)]
            else:
                pre_sup = tf.matmul(x, vars['weights_{}'.format(i)])
            support = tf.sparse_tensor_dense_matmul(support[i], pre_sup)
            supports.append(support)
        output = tf.add_n(supports)
    return output, vars,


def sparse_dropout(x, keep_pro, noise_shape):
    random_tensor = keep_pro + tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    x = tf.sparse_retain(x, dropout_mask)
    return x * (1 / keep_pro)
