import numpy as np
from collections import Counter
from pathlib import Path

from utils import yaml_utils

vocabulary_size = 7000
dataset_name = 'cnews'
output_dir = Path('../dataset').absolute()


def read_data(path):
    print('导入数据')
    data_path = Path(path)
    contents, labels = [], []
    with data_path.open(mode='r', encoding='UTF-8') as file:
        for line in file:
            label, content = line.strip().split('\t')
            if content:
                contents.append(content)
                labels.append(label)
    print('导入完成')
    return contents, labels


def build_dataset(data):
    """根据训练集构建词汇表，存储"""
    print('制作词典')
    all_words = []
    for line in data[0]:
        all_words.extend(line)
    count = Counter(all_words).most_common(vocabulary_size - 1)
    words, _ = list(zip(*count))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    dictionary = dict(zip(words, range(len(words))))
    print('导出字典,训练和验证集')
    print('制作反向查询词典')
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    print('制作反向查询词典完成')
    output_path = output_dir / dataset_name
    dataset_dict = list()
    data_len = len(data[0])
    for i in range(data_len):
        dataset_dict.append({'input': data[0][i], 'label': data[1][i]})
    np.random.shuffle(dataset_dict)
    train_dict = dataset_dict[:int(data_len * 0.9)]
    eval_dict = dataset_dict[int(data_len * 0.9):]
    yaml_utils.write(output_path / 'train.yaml', train_dict)
    yaml_utils.write(output_path / 'eval.yaml', eval_dict)
    labels = set(data[1])
    labels_dictionary = dict(zip(labels, range(len(labels))))
    reverse_labels_dictionary = dict(zip(labels_dictionary.values(), labels_dictionary.keys()))
    yaml_utils.write(output_path / 'dictionary.yaml',
                     {'word_dictionary': dictionary, 'label_dictionary': labels_dictionary})
    yaml_utils.write(output_path / 'reverse_dictionary.yaml',
                     {'word_dictionary': reverse_dictionary, 'label_dictionary': reverse_labels_dictionary})

    yaml_utils.write(output_path / 'info.yaml', {'vocabulary_size': vocabulary_size, 'num_class': len(labels),
                                                 'train_path': str(output_path / 'train.yaml'),
                                                 'eval_path': str(output_path / 'eval.yaml'),
                                                 'dictionary_path': str(output_path / 'dictionary.yaml'),
                                                 'reverse_dictionary_path': str(
                                                     output_path / 'reverse_dictionary.yaml')})
    print('导出字典,训练和验证集完成')


if __name__ == '__main__':
    data = read_data('../data/{}/{}.txt'.format(dataset_name, dataset_name))
    # if not (output_dir / dataset_name / 'dictionary.yaml').exists():
    build_dataset(data)
