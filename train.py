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
                            **dataset_info, **args['dataset'], **args['model'], **args)
        model.train()


if __name__ == '__main__':
    config = get_config('rnn/cnews')
    config['tag'] = 'base'
    train(config)
