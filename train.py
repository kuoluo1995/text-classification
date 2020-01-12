import os

import numpy as np
import tensorflow as tf

from data_loader import get_data_loader_by_name
from models import get_model_class_by_name
from utils import yaml_utils
from utils.config_utils import get_config


def train(args):
    tf.set_random_seed(19)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        print('导入数据')
        dataset_info = yaml_utils.read(args['dataset']['path'])
        dictionary = yaml_utils.read(dataset_info['dictionary_path'])
        print('导入数据字典完成')
        if 'embedding_path' in dataset_info.keys():
            embedding = np.array(yaml_utils.read(dataset_info['embedding_path']), dtype=np.float32)
        else:
            embedding = None
        print('导入词向量完成')
        train_dataset = yaml_utils.read(dataset_info['train_path'])
        print('导入数据训练数据完成')
        eval_dataset = yaml_utils.read(dataset_info['eval_path'])
        print('导入数据验证数据完成')
        print('导入完成')
        data_loader = get_data_loader_by_name(args['dataset']['data_generator'])
        train_data_generator = data_loader(dictionary, True, train_dataset, batch_size=args['batch_size'],
                                           seq_length=args['dataset']['seq_length'])

        eval_data_generator = data_loader(dictionary, False, eval_dataset, batch_size=args['batch_size'],
                                          seq_length=args['dataset']['seq_length'])
        model_class = get_model_class_by_name(args['model']['name'])
        model = model_class(sess=sess, train_generator=train_data_generator, eval_generator=eval_data_generator,
                            embedding=embedding, **dataset_info, **args['dataset'], **args['model'], **args)
        model.train()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    # config = get_config('adversarial/aclImdb_rnn')
    # config = get_config('adversarial/aclImdb_cnn')
    # config = get_config('adversarial/cnews_rnn')
    # config = get_config('adversarial/cnews_cnn')
    # config = get_config('cnn/aclImdb')
    # config = get_config('cnn/cnews')
    # config = get_config('rnn/aclImdb')
    config = get_config('rnn/cnews')
    config['tag'] = 'base'
    # config['model']['num_layers'] = 3
    train(config)
