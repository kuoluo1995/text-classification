import tensorflow as tf


def gcn_layer(x, adjacency_matrix, in_channels, out_channels, keep_pro, is_sparse, num_nonzero=None, name='gcn_layer'):
    with tf.name_scope(name):
        with tf.variable_scope('{}_vars'.format(name)):
            weights = tf.get_variable('weights', [in_channels, out_channels],
                                      initializer=tf.initializers.glorot_normal())
        if is_sparse:
            x = sparse_dropout(x, keep_pro, num_nonzero)
            x = tf.sparse_tensor_dense_matmul(x, weights)
        else:
            x = tf.nn.dropout(x, keep_pro)
            x = tf.matmul(x, weights)
        support = tf.sparse_tensor_dense_matmul(adjacency_matrix, x)
    return support, weights


def sparse_dropout(x, keep_pro, noise_shape):
    random_tensor = keep_pro + tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    x = tf.sparse_retain(x, dropout_mask)
    return x * (1 / keep_pro)
