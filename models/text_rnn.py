import tensorflow as tf

from models.text_cnn import TextCNN


class TextRNN(TextCNN):
    def __init__(self, rnn_type, **kwargs):
        self.rnn_type = rnn_type
        TextCNN.__init__(self, **kwargs)

    def lstm_cell(self, reuse):  # lstm核
        return tf.nn.rnn_cell.LSTMCell(self.num_filters, forget_bias=0.0, reuse=reuse)

    def gru_cell(self, reuse):  # gru核
        return tf.nn.rnn_cell.GRUCell(self.num_filters, reuse=reuse)

    def dropout(self, reuse=False):  # 为每一个rnn核后面加一个dropout层
        cell = self.gru_cell(reuse) if self.rnn_type == 'gru' else self.lstm_cell(reuse)
        return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob_tensor)

    def get_logits(self, inputs, reuse=False):
        with tf.variable_scope('get_logits', reuse=reuse):
            embedding_inputs = tf.nn.embedding_lookup(self.embedding, inputs)
            with tf.name_scope('rnn'):
                # 多层rnn网络
                cells = [self.dropout() for _ in range(self.num_layers)]
                rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
                _outputs, next_state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
                last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果

            with tf.name_scope('score'):
                # 全连接层，后面接dropout以及relu激活
                fc = tf.layers.dense(last, self.hidden_size, name='fc1')
                fc = tf.nn.relu(fc)
                # 分类器
                logits = tf.layers.dense(fc, self.num_class, name='fc2')
            return logits
