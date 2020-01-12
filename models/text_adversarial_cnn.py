import tensorflow as tf

from models.text_adversarial_rnn import TextAdversarialRNN


class TextAdversarialCNN(TextAdversarialRNN):
    def __init__(self, **kwargs):
        TextAdversarialRNN.__init__(self, **kwargs)

    def build_networks(self):
        # Input data.
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, [None, self.seq_length], name='inputs')
            self.labels = tf.placeholder(tf.float32, [None, self.num_class], name='labels')
            self.keep_prob_tensor = tf.placeholder(tf.float32, name='keep_prob')
        # 词向量映射
        if self.init_embedding is not None:
            self.embedding = tf.get_variable('embedding', initializer=self.init_embedding, dtype=tf.float32)
        else:
            self.embedding = tf.get_variable('embedding', [self.vocabulary_size, self.embedding_dim], dtype=tf.float32)

        logits, embedding_inputs = self.get_logits(self.inputs)
        # loss
        with tf.name_scope('loss'):
            cl_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.labels))
        with tf.name_scope('adversarial'):
            perturbed = self.get_adversarial(cl_loss, embedding_inputs)
        _logits, _ = self.get_logits(self.inputs, perturbed, reuse=True)
        with tf.name_scope('loss'):
            adv_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=_logits, labels=self.labels))
            self.loss = cl_loss + adv_loss

        with tf.name_scope('accuracy'):
            self.target = tf.argmax(tf.nn.softmax(logits), 1)  # 预测类别
            correct_pred = tf.equal(tf.argmax(self.labels, 1), self.target)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        self.build_summary()

    def get_logits(self, inputs, perturbed=None, reuse=False):
        with tf.variable_scope('get_logits', reuse=reuse):
            embedding_inputs = tf.nn.embedding_lookup(self.embedding, inputs)
            if perturbed is not None:
                embedding_inputs += perturbed
            inputs = embedding_inputs
            with tf.name_scope('cnn'):
                for i in range(self.num_layers):
                    # CNN layer
                    inputs = tf.layers.conv1d(inputs, self.num_filters, self.kernel_size, name='conv_{}'.format(i))
                    # global max pooling layer
                    inputs = tf.reduce_max(inputs, reduction_indices=[1], name='gmp')

            with tf.name_scope('score'):
                # 全连接层，后面接dropout以及relu激活
                fc = tf.layers.dense(inputs, self.hidden_size, name='fc1')
                fc = tf.layers.dropout(fc, self.keep_prob_tensor)
                fc = tf.nn.relu(fc)
                # 分类器
                logits = tf.layers.dense(fc, self.num_class, name='fc2')
            return logits, embedding_inputs

    def get_adversarial(self, loss, embedding_inputs, perturbed=0.02):  # Perturbation MULTiplier
        gradient = tf.gradients(loss, embedding_inputs,
                                aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        perturbed_adversarial = perturbed * tf.nn.l2_normalize(tf.stop_gradient(gradient[0]), axis=1)
        return perturbed_adversarial
