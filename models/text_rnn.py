import tensorflow as tf
from models.text_cnn import TextCNN


class TextRNN(TextCNN):
    def __init__(self, rnn_type, num_layers, **kwargs):
        self.num_layers = num_layers
        self.num_filters = kwargs['num_filters']
        self.rnn_type = rnn_type
        TextCNN.__init__(self, **kwargs)

    def lstm_cell(self):  # lstm核
        return tf.nn.rnn_cell.BasicLSTMCell(self.num_filters, state_is_tuple=True)

    def gru_cell(self):  # gru核
        return tf.nn.rnn_cell.GRUCell(self.num_filters)

    def dropout(self):  # 为每一个rnn核后面加一个dropout层
        cell = self.gru_cell() if self.rnn_type == 'gru' else self.lstm_cell()
        return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

    def build_networks(self):
        # Input data.
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, [None, self.seq_length], name='inputs')
            self.labels = tf.placeholder(tf.float32, [None, self.num_class], name='labels')
            self.keep_prob_tensor = tf.placeholder(tf.float32, name='keep_prob')
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.vocabulary_size, self.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.inputs)
        with tf.name_scope('rnn'):
            # 多层rnn网络
            cells = [self.dropout() for _ in range(self.num_layers)]
            rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果
        with tf.name_scope('score'):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, self.num_filters, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob_tensor)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.num_class, name='fc2')
            self.target = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.labels))
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.labels, 1), self.target)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.name_scope('tensorboard'):
            loss_scalar = tf.summary.scalar('train_loss', self.loss)
            accuracy_scalar = tf.summary.scalar('train_accuracy', self.accuracy)
            self.eval_loss = tf.placeholder(tf.float32, name='eval_loss')
            eval_loss_scalar = tf.summary.scalar('loss', self.eval_loss)
            self.eval_accuracy = tf.placeholder(tf.float32, name='eval_accuracy')
            eval_accuracy_scalar = tf.summary.scalar('accuracy', self.eval_accuracy)
            self.scalar_summary = tf.summary.merge(
                [loss_scalar, accuracy_scalar, eval_loss_scalar, eval_accuracy_scalar])
