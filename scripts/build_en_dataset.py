import re

import numpy as np
from collections import Counter
from pathlib import Path

from utils import yaml_utils

vocabulary_size = 14000
dataset_name = 'aclImdb'
output_dir = Path('../dataset').absolute()
output_path = output_dir / dataset_name


def read_data(path):
    data_path = Path(path)
    contents = []
    with data_path.open(mode='r', encoding='UTF-8') as file:
        for line in file:
            while '\n' in line:
                line = line.replace('\n', '')
            while '<br />' in line:
                line = line.replace('<br />', '')
            if len(line) > 0:
                contents.append(line)
    return contents


def build_dictionary(data, types):
    print('制作词典')
    all_words = []
    for line in data:
        all_words.extend(line.split())
    count = Counter(all_words).most_common(vocabulary_size - 1)
    words, _ = list(zip(*count))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    dictionary = dict(zip(words, range(len(words))))

    labels = set(types)
    labels_dictionary = dict(zip(labels, range(len(labels))))
    print('制作反向查询词典')

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    reverse_labels_dictionary = dict(zip(labels_dictionary.values(), labels_dictionary.keys()))
    print('制作反向查询词典完成')
    yaml_utils.write(output_path / 'dictionary.yaml',
                     {'word_dictionary': dictionary, 'label_dictionary': labels_dictionary})
    yaml_utils.write(output_path / 'reverse_dictionary.yaml',
                     {'word_dictionary': reverse_dictionary, 'label_dictionary': reverse_labels_dictionary})


def build_dataset(dataset, num_class):
    """根据训练集构建词汇表，存储"""
    print('创建训练和验证集')
    dataset_len = len(dataset)
    np.random.shuffle(dataset)
    train_dict = dataset[:int(dataset_len * 0.9)]
    eval_dict = dataset[int(dataset_len * 0.9):]
    yaml_utils.write(output_path / 'train.yaml', train_dict)
    yaml_utils.write(output_path / 'eval.yaml', eval_dict)

    yaml_utils.write(output_path / 'info.yaml', {'vocabulary_size': vocabulary_size, 'num_class': num_class,
                                                 'train_path': str(output_path / 'train.yaml'),
                                                 'eval_path': str(output_path / 'eval.yaml'),
                                                 'dictionary_path': str(output_path / 'dictionary.yaml'),
                                                 'reverse_dictionary_path': str(
                                                     output_path / 'reverse_dictionary.yaml')})
    print('创建训练和验证集完成')


if __name__ == '__main__':
    dataset = list()
    contents = list()
    class_type = list()
    dataset_fold = Path('../data/{}'.format(dataset_name)).absolute()
    for type in dataset_fold.iterdir():
        class_type.append(type.name)
        for item in type.iterdir():
            content = read_data(str(item))
            contents.extend(content)
            dataset.append({'input': str(item), 'label': type.name})
    build_dictionary(contents, class_type)
    build_dataset(dataset, len(class_type))
