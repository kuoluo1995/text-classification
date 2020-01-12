import re
from collections import Counter

import numpy as np
from pathlib import Path

from utils import yaml_utils

dataset_name = 'aclImdb'
# E:/Dataset/aclImdb/glove.6B.300d.txt /home/yf/dataset/glove.6B.300d.txt
embedding_file = 'E:/Dataset/aclImdb/glove.6B.300d.txt'
embedding_dim = 300  # 词向量维度
max_vocabulary_size = 10 * 1000
# /home/yf/dataset/{}/train E:/Dataset/{}/train
dataset_fold = Path('E:/Dataset/{}/train'.format(dataset_name)).absolute()
output_dir = Path('../dataset').absolute()
output_path = output_dir / dataset_name
output_path.mkdir(exist_ok=True, parents=True)


def loadGloVe(filename, embedding_dim):
    vocab = list()
    embedding = list()
    vocab.append('<pad>')  # 装载不认识的词和空余词
    embedding.append([0] * embedding_dim)
    file = open(filename, 'r', encoding='UTF-8')
    for i, line in enumerate(file.readlines()):
        if i >= max_vocabulary_size:
            break
        row = line.strip().split(' ')
        vocab.append(row[0])
        embedding.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    yaml_utils.write(output_path / 'embedding.yaml', embedding)
    return vocab, len(vocab)


def build_vocabulary(contents):
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    vocabulary = []
    for line in contents:
        vocabulary += read_data(line['input'])

    count = Counter(vocabulary).most_common(max_vocabulary_size)
    vocabulary, _ = list(zip(*count))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    vocabulary = ['<pad>'] + list(vocabulary)
    return vocabulary, len(vocabulary)


def read_data(path):
    data_path = Path(path)
    contents = []
    with data_path.open(mode='r', encoding='UTF-8') as file:
        for line in file:
            while '\n' in line:
                line = line.replace('\n', '')
            # 之前发现训练结果不好，然后发现原来时由于英文数据符合断句问题导致的
            while '<br />' in line:
                line = line.replace('<br />', '')
            words = [s for s in re.split(r'\W+', line) if s and not s.isspace()]
            if len(line) > 0:
                contents.extend(words)
    return contents


def build_dictionary(vocabulary, labels):
    print('制作词典')
    dictionary = dict(zip(vocabulary, range(len(vocabulary))))

    labels_dictionary = dict(zip(labels, range(len(labels))))
    print('制作反向查询词典')

    reverse_dictionary = dict()
    for key, item in dictionary.items():
        reverse_dictionary[item] = key
    reverse_labels_dictionary = dict()
    for key, item in labels_dictionary.items():
        reverse_labels_dictionary[item] = key
    print('制作反向查询词典完成')
    yaml_utils.write(output_path / 'dictionary.yaml',
                     {'word_dictionary': dictionary, 'label_dictionary': labels_dictionary})
    yaml_utils.write(output_path / 'reverse_dictionary.yaml',
                     {'word_dictionary': reverse_dictionary, 'label_dictionary': reverse_labels_dictionary})


def build_dataset(dataset, num_class, vocabulary_size, has_embedding=False):
    """根据训练集构建词汇表，存储"""
    print('创建训练和验证集')
    dataset_len = len(dataset)
    np.random.shuffle(dataset)
    train_dict = dataset[:int(dataset_len * 0.9)]
    eval_dict = dataset[int(dataset_len * 0.9):]
    yaml_utils.write(output_path / 'train.yaml', train_dict)
    yaml_utils.write(output_path / 'eval.yaml', eval_dict)
    info = {'vocabulary_size': vocabulary_size, 'num_class': num_class, 'embedding_dim': embedding_dim,
            'train_path': str(output_path / 'train.yaml'), 'eval_path': str(output_path / 'eval.yaml'),
            'dictionary_path': str(output_path / 'dictionary.yaml'),
            'reverse_dictionary_path': str(output_path / 'reverse_dictionary.yaml')}
    if has_embedding:
        info.update({'embedding_path': str(output_path / 'embedding.yaml')})
    yaml_utils.write(output_path / 'info.yaml', info)
    print('导出字典,训练和验证集完成')


if __name__ == '__main__':
    dataset = list()
    class_type = set()
    for type in dataset_fold.iterdir():
        if type.is_dir() and type.name != 'unsup':
            class_type.add(type.name)
            for item in type.iterdir():
                dataset.append({'input': str(item), 'label': type.name})
    # vocabulary, vocabulary_size = build_vocabulary(dataset)
    vocabulary, vocabulary_size = loadGloVe(embedding_file, embedding_dim)
    build_dictionary(vocabulary, class_type)
    build_dataset(dataset, len(class_type), vocabulary_size, has_embedding=True)
