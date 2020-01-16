import os
import tensorflow as tf
from data_loader.gnn_data_generator import load_data
from models.text_gnn import TextGNN
from utils.config_utils import get_config


def train(args):
    tf.set_random_seed(19)
    tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=tf_config) as sess:
        data_generator = load_data(args['dataset']['path'], args['dataset']['dataset_name'])
        model = TextGNN(sess=sess, data_generator=data_generator, **args['dataset'], **args['model'], **args)
        model.train()


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    config = get_config('gnn/aclImdb')
    config['tag'] = 'my_dataset2'
    train(config)
