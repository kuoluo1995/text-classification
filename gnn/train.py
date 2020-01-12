import numpy as np
import tensorflow as tf

from data_loader import get_data_loader_by_name
from gnn.load_data import load_network
from gnn.text_gnn import TextGNN
from models import get_model_class_by_name
from utils import yaml_utils
from utils.config_utils import get_config


def train(args):
    tf.set_random_seed(19)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        print('导入数据')
        print('导入词向量完成')
        data_generator = load_network('E:/Dataset/gnn', 'cora')
        model = TextGNN(sess=sess, train_generator=data_generator, **args['dataset'], **args['model'], **args)
        # model_class = get_model_class_by_name(args['model']['name'])
        # model = model_class(sess=sess, train_generator=train_data_generator, eval_generator=None, embedding=embedding,
        #                     **dataset_info, **args['dataset'], **args['model'], **args)
        model.train()


if __name__ == "__main__":
    config = get_config('gnn/cora')
    config['tag'] = 'base'
    # config['model']['num_layers'] = 3
    train(config)
