import numpy as np
import tensorflow as tf
from sklearn import metrics

from data_loader import get_data_loader_by_name
from models import get_model_class_by_name
from utils import yaml_utils
from utils.config_utils import get_config


def evaluate_model(args):
    tf.set_random_seed(19)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        print('导入数据')
        dataset_info = yaml_utils.read(args['dataset']['path'])
        dictionary = yaml_utils.read(dataset_info['dictionary_path'])
        print('导入数据字典完成')
        reverse_dictionary = yaml_utils.read(dataset_info['reverse_dictionary_path'])
        print('导入反向字典完成')
        if 'embedding_path' in dataset_info.keys():
            embedding = np.array(yaml_utils.read(dataset_info['embedding_path']), dtype=np.float32)
        else:
            embedding = None
        print('导入词向量完成')
        test_dataset = yaml_utils.read(dataset_info['eval_path'])
        print('导入测试数据完成')
        print('导入完成')
        data_loader = get_data_loader_by_name(args['dataset']['data_generator'])
        eval_data_generator = data_loader(dictionary, False, test_dataset, batch_size=args['batch_size'],
                                          seq_length=args['dataset']['seq_length'],
                                          reverse_dictionary=reverse_dictionary)
        eval_data_generator.get_reverse_dictionary()
        model_class = get_model_class_by_name(args['model']['name'])
        model = model_class(sess=sess, train_generator=None, eval_generator=eval_data_generator,
                            embedding=embedding, **dataset_info, **args['dataset'], **args['model'], **args)
        result, labels = model.test()
        # yaml_utils.write(args['model']['checkpoint_dir'] + '/' + args['dataset']['dataset_name'] + '/' +
        #                  args['model']['name'] + '/' + args['tag'] + '/' + 'best_result.yaml', result)
        print('评估')
        print(metrics.classification_report(labels, result, target_names=eval_data_generator.get_labels()))
        print('混淆矩阵')
        cm = metrics.confusion_matrix(labels, result)
        print(cm)


if __name__ == '__main__':
    config = get_config('adversarial/aclImdb_rnn')
    # config = get_config('adversarial/aclImdb_cnn')
    # config = get_config('adversarial/cnews_rnn')
    # config = get_config('adversarial/cnews_cnn')
    # config = get_config('cnn/aclImdb')
    # config = get_config('cnn/cnews')
    # config = get_config('rnn/aclImdb')
    # config = get_config('rnn/cnews')
    config['tag'] = 'base'
    # config['model']['num_layers'] = 3
    evaluate_model(config)
