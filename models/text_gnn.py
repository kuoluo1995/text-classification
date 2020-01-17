import numpy as np
import tensorflow as tf
from models.networks import gcn_layer
from pathlib import Path


class TextGNN:
    def __init__(self, sess, tag, save_freq, num_epochs, dataset_name, input_shape, num_class, num_support,
                 data_generator, name, checkpoint_dir, num_hidden, keep_pro, learning_rate, weight_decay, **kwargs):
        self.sess = sess
        self.tag = tag
        self.num_epochs = num_epochs
        self.save_freq = save_freq

        self.dataset_name = dataset_name
        self.data_generator = data_generator
        self.input_shape = input_shape
        self.num_class = num_class
        self.num_support = num_support

        self.name = name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.num_hidden = num_hidden
        self.keep_pro = keep_pro
        self.learning_rate = learning_rate
        self.weight_decay = np.float(weight_decay)
        self.build_networks()
        self.train_saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()

    def build_networks(self):
        # Input data.
        with tf.name_scope('inputs'):
            # Define Placeholders
            self.support = [tf.sparse_placeholder(tf.float32, name='support_{}'.format(i)) for i in
                            range(self.num_support)]
            self.features = tf.sparse_placeholder(tf.float32, shape=self.input_shape, name='features')
            self.labels = tf.placeholder(tf.float32, shape=[None, self.num_class], name='labels')
            self.labels_mask = tf.placeholder(tf.int32, name='labels_mask')
            self.keep_pro_tensor = tf.placeholder(tf.float32, name='keep_pro')
            self.num_nonzero = tf.placeholder(tf.int32, name='num_nonzero')

        with tf.name_scope('gnn'):
            # Construct Computational Graph
            gcn1, vars1 = gcn_layer(self.features, self.support, in_channels=self.input_shape[-1],
                                    out_channels=self.num_hidden, keep_pro=self.keep_pro_tensor, is_sparse=True,
                                    num_nonzero=self.num_nonzero, name='gcn_layer1')
            gcn1 = tf.nn.relu(gcn1)

            gcn2, vars2 = gcn_layer(gcn1, self.support, in_channels=self.num_hidden, out_channels=self.num_class,
                                    keep_pro=self.keep_pro_tensor, is_sparse=False, name='gcn_layer2')

        mask = tf.cast(self.labels_mask, dtype=tf.float32)  # Cast masking from boolean to float
        mask /= tf.reduce_mean(mask)  # Compute mean for mask

        with tf.name_scope('loss'):
            # Compute cross entropy loss
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=gcn2, labels=self.labels)
            loss *= mask  # Mask the output of cross entropy loss
            loss = tf.reduce_mean(loss)
            for _, var in vars1.items():
                loss += self.weight_decay * tf.nn.l2_loss(var)
            for _, var in vars2.items():
                loss += self.weight_decay * tf.nn.l2_loss(var)
            self.loss = loss

        with tf.name_scope('accuracy'):
            # Identity position where prediction matches labels
            self.target = tf.argmax(gcn2, 1)
            self.label = tf.argmax(self.labels, 1)
            correct_pred = tf.equal(self.target, self.label)
            # Cast result to float
            accuracy = tf.cast(correct_pred, tf.float32)
            accuracy *= mask  # Apply mask on computed accuracy
            self.accuracy = tf.reduce_mean(accuracy)
        self.build_summary()

    def build_summary(self):
        with tf.name_scope('tensorboard'):
            loss_scalar = tf.summary.scalar('train_loss', self.loss)
            accuracy_scalar = tf.summary.scalar('train_accuracy', self.accuracy)
            self.eval_loss = tf.placeholder(tf.float32, name='eval_loss')
            eval_loss_scalar = tf.summary.scalar('eval_loss', self.eval_loss)
            self.eval_accuracy = tf.placeholder(tf.float32, name='eval_accuracy')
            eval_accuracy_scalar = tf.summary.scalar('eval_accuracy', self.eval_accuracy)
            self.scalar_summary = tf.summary.merge(
                [loss_scalar, accuracy_scalar, eval_loss_scalar, eval_accuracy_scalar])

    def train(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        writer = tf.summary.FileWriter('../tensorboard_logs/{}/{}/{}'.format(self.dataset_name, self.name, self.tag),
                                       self.sess.graph)
        best_glob_accuracy = 0
        print('开始训练')
        for epoch in range(self.num_epochs):
            # Training step
            feed_dict = {self.features: self.data_generator['features'],
                         self.num_nonzero: self.data_generator['num_nonzero']}
            feed_dict.update({self.support[i]: self.data_generator['support'][i] for i in range(self.num_support)})
            feed_dict.update({self.labels: self.data_generator['y_train']})
            feed_dict.update({self.labels_mask: self.data_generator['mask_train']})
            feed_dict.update({self.keep_pro_tensor: self.keep_pro})
            _, train_loss, train_accuracy = self.sess.run([optimizer, self.loss, self.accuracy], feed_dict=feed_dict)

            # Evaling step
            feed_dict.update({self.labels: self.data_generator['y_test']})
            feed_dict.update({self.labels_mask: self.data_generator['mask_test']})
            feed_dict.update({self.keep_pro_tensor: 1.0})
            eval_loss, eval_accuracy = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
            print('第{}轮训练：train_loss:{},train_accuracy:{},eval_loss:{},eval_accuracy:{}'
                  .format(epoch + 1, train_loss, train_accuracy, eval_loss, eval_accuracy))
            summary = self.sess.run(self.scalar_summary,
                                    feed_dict={self.loss: train_loss, self.accuracy: train_accuracy,
                                               self.eval_loss: eval_loss, self.eval_accuracy: eval_accuracy})
            if epoch % self.save_freq == 0:
                writer.add_summary(summary, epoch // self.save_freq)
                self.save(self.checkpoint_dir / 'train', self.train_saver, epoch)
            if best_glob_accuracy < (train_accuracy + eval_accuracy):
                self.save(self.checkpoint_dir / 'best', self.best_saver, epoch)
                best_glob_accuracy = train_accuracy + eval_accuracy

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

    def test(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.load(self.checkpoint_dir / 'best', self.best_saver)
        print('开始测试')
        feed_dict = dict()
        feed_dict.update({self.features: self.data_generator['features']})
        feed_dict.update(
            {self.support[i]: self.data_generator['support'][i] for i in range(self.data_generator['num_support'])})
        feed_dict.update({self.num_nonzero: self.data_generator['num_nonzero']})
        feed_dict.update({self.labels: self.data_generator['y_test']})
        feed_dict.update({self.labels_mask: self.data_generator['mask_test']})
        feed_dict.update({self.keep_pro_tensor: 1.0})
        result = list()
        labels = list()
        result_, labels_, accuarcy = self.sess.run([self.target, self.label, self.accuracy], feed_dict=feed_dict)
        for i in range(len(self.data_generator['mask_test'])):
            if self.data_generator['mask_test'][i]:
                result.append(result_[i])
                labels.append(labels_[i])
        return result, labels
