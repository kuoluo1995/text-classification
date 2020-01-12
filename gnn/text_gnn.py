import tensorflow as tf

from gnn.networks import gcn_layer
from models.text_cnn import TextCNN


class TextGNN:
    def __init__(self, sess, tag, save_freq, num_epochs, dataset_name, train_generator, seq_length, num_class, name,
                 checkpoint_dir, num_filters, keep_prob, learning_rate, num_node, l2, **kwargs):
        self.sess = sess
        self.tag = tag
        self.num_epochs = num_epochs
        self.save_freq = save_freq

        self.dataset_name = dataset_name
        self.train_generator = train_generator
        self.seq_length = seq_length
        self.num_class = num_class

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.num_node = num_node
        self.num_filters = num_filters
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate
        self.l2 = l2

    def build_networks(self):
        # Input data.
        with tf.name_scope('inputs'):
            # Define Placeholders
            self.features = tf.sparse_placeholder(tf.float32, shape=[self.num_node, self.seq_length], name='features')
            self.adjacency_matrix = tf.sparse_placeholder(tf.float32, shape=[self.num_node, self.num_node],
                                                          name='support')
            self.labels = tf.placeholder(tf.float32, shape=[None, self.num_class], name='labels')
            self.labels_mask = tf.placeholder(tf.int32, name='labels_mask')
            self.keep_prob_tensor = tf.placeholder(tf.float32, name='keep_prob')
            self.num_nonzero = tf.placeholder(tf.int32, name='num_nonzero')

        with tf.name_scope('gnn'):
            # Construct Computational Graph
            gcn1, vars1 = gcn_layer(self.features, self.seq_length, self.num_filters, self.adjacency_matrix,
                                    self.keep_prob_tensor, True, self.num_nonzero, name='gcn_layer1')
            gcn1 = tf.nn.relu(gcn1)

            gcn2, vars2 = gcn_layer(gcn1, self.num_filters, self.num_class, self.adjacency_matrix,
                                    self.keep_prob_tensor, False, name='gcn_layer2')

        mask = tf.cast(self.labels_mask, dtype=tf.float32)  # Cast masking from boolean to float todo check?
        mask /= tf.reduce_mean(mask)  # Compute mean for mask

        with tf.name_scope('loss'):
            # Compute cross entropy loss
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=gcn2, labels=self.labels)
            loss *= mask  # Mask the output of cross entropy loss
            loss = tf.reduce_mean(loss)
            self.loss = loss + self.l2 * (tf.nn.l2_loss(vars1) + tf.nn.l2_loss(vars2))

        with tf.name_scope('accuracy'):
            # Identity position where prediction matches labels
            correct_pred = tf.equal(tf.argmax(gcn2, 1), tf.argmax(self.labels, 1))
            # Cast result to float
            accuracy = tf.cast(correct_pred, tf.float32)
            accuracy *= mask  # Apply mask on computed accuracy
            self.accuracy = tf.reduce_mean(accuracy)

        with tf.name_scope('tensorboard'):
            loss_scalar = tf.summary.scalar('train_loss', self.loss)
            accuracy_scalar = tf.summary.scalar('train_accuracy', self.accuracy)
            self.eval_loss = tf.placeholder(tf.float32, name='eval_loss')
            eval_loss_scalar = tf.summary.scalar('loss', self.eval_loss)
            self.eval_accuracy = tf.placeholder(tf.float32, name='eval_accuracy')
            eval_accuracy_scalar = tf.summary.scalar('accuracy', self.eval_accuracy)
            self.scalar_summary = tf.summary.merge(
                [loss_scalar, accuracy_scalar, eval_loss_scalar, eval_accuracy_scalar])

    def train(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        writer = tf.summary.FileWriter('../tensorboard_logs/{}/{}/{}'.format(self.dataset_name, self.name, self.tag),
                                       self.sess.graph)
        best_glob_accuracy = 0
        train_size = self.train_generator.get_size()
        feature, adjacency_matrix, num_nonzero, train_labels, train_labels_mask, eval_labels, eval_labels_mask = self.train_generator
        print('开始训练')
        for epoch in range(self.num_epochs):
            train_loss = 0
            train_accuracy = 0
            for step in range(train_size):
                # Training step
                _, loss, accuracy = self.sess.run([optimizer, self.loss, self.accuracy],
                                                  feed_dict={self.features: feature,
                                                             self.adjacency_matrix: adjacency_matrix,
                                                             self.num_nonzero: num_nonzero, self.labels: train_labels,
                                                             self.labels_mask: train_labels_mask,
                                                             self.keep_prob_tensor: self.keep_prob})
                train_loss += loss
                train_accuracy += accuracy

            print('第{}轮训练：train_loss:{},train_accuracy:{},eval_loss:{},eval_accuracy:{}'
                  .format(epoch + 1, train_loss / train_size, train_accuracy / train_size, 0, 0))
            summary = self.sess.run(self.scalar_summary,
                                    feed_dict={self.loss: train_loss / train_size,
                                               self.accuracy: train_accuracy / train_size,
                                               self.eval_loss: 0,
                                               self.eval_accuracy: 0})
            writer.add_summary(summary, epoch)
            # save train model
            # if epoch % self.save_freq == 0:
            #     self.save(self.checkpoint_dir / 'train', self.train_saver, epoch)
            # if best_glob_accuracy < (train_accuracy + eval_accuracy):
            #     self.save(self.checkpoint_dir / 'best', self.best_saver, epoch)
            #     best_glob_accuracy = train_accuracy + eval_accuracy

    def save(self, checkpoint_dir, saver, epoch, **kwargs):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        saver.save(self.sess, str(checkpoint_dir / 'model.cpk'), global_step=epoch)

    def load(self, checkpoint_dir, saver, **kwargs):
        # checkpoint = tf.train.get_checkpoint_state(str(checkpoint_dir))
        checkpoint = tf.train.latest_checkpoint(str(checkpoint_dir))
        if checkpoint:
            # saver.restore(self.sess, checkpoint.model_checkpoint_path)
            saver.restore(self.sess, checkpoint)
        else:
            print('Loading checkpoint failed')
