import tensorflow as tf
from sklearn import metrics
from data_loader.gnn_data_generator import load_data
from models.text_gnn import TextGNN
from utils import csv_utils
from utils.config_utils import get_config


def evaluate_model(args):
    tf.set_random_seed(19)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        label_dict = csv_utils.read(args['dataset']['path'] + '/labels.csv')
        data_generator = load_data(args['dataset']['path'], args['dataset']['dataset_name'])
        model = TextGNN(sess=sess, data_generator=data_generator, **data_generator, **args['dataset'], **args['model'],
                        **args)
        result, labels = model.test()

        print('评估')
        print(metrics.classification_report(labels, result, target_names=label_dict))
        print('混淆矩阵')
        print(metrics.confusion_matrix(labels, result))


if __name__ == '__main__':
    # config = get_config('gnn/aclImdb')
    # config = get_config('gnn/cnews')
    config = get_config('gnn/cnews_voc')
    config['tag'] = 'base'
    evaluate_model(config)
