import yaml
import jieba
import numpy as np
from collections import Counter
from pathlib import Path

dataset_name = 'cnews_voc'
embedding_file = None
embedding_dim = 64  # 词向量维度
max_vocabulary_size = 7000
dataset_file = Path('../data/cnews/cnews.txt')
output_dir = Path('../dataset').absolute()
output_path = output_dir / dataset_name
output_path.mkdir(exist_ok=True, parents=True)


def yaml_write(path, data, encoding='utf-8'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding=encoding) as file:
        yaml.dump(data, file, allow_unicode=True)


def read_data(path):
    print('导入数据')
    data_path = Path(path)
    dataset = list()
    labels = set()
    with data_path.open(mode='r', encoding='UTF-8') as file:
        for line in file:
            label, content = line.strip().split('\t')
            if content:
                while '\n' in content:
                    content = content.replace('\n', '')
                while ' ' in content:
                    content = content.replace(' ', '')
                words = list(jieba.cut(content, cut_all=False))
                labels.add(label)
                dataset.append({'input': words, 'label': label})
    print('导入完成')
    return dataset, labels


def build_vocabulary(contents):
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    vocabulary = []
    for line in contents:
        vocabulary += line['input']
    count = Counter(vocabulary).most_common(max_vocabulary_size)
    vocabulary, _ = list(zip(*count))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    vocabulary = ['<pad>'] + list(vocabulary)
    return vocabulary, len(vocabulary)


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
    yaml_write(output_path / 'dictionary.yaml',
               {'word_dictionary': dictionary, 'label_dictionary': labels_dictionary})
    yaml_write(output_path / 'reverse_dictionary.yaml',
               {'word_dictionary': reverse_dictionary, 'label_dictionary': reverse_labels_dictionary})


def build_dataset(dataset, num_class, vocabulary_size):
    """根据训练集构建词汇表，存储"""
    print('创建训练和验证集')
    dataset_len = len(dataset)
    np.random.shuffle(dataset)
    train_dict = dataset[:int(dataset_len * 0.9)]
    eval_dict = dataset[int(dataset_len * 0.9):]
    yaml_write(output_path / 'train.yaml', train_dict)
    yaml_write(output_path / 'eval.yaml', eval_dict)
    info = {'vocabulary_size': vocabulary_size, 'num_class': num_class, 'embedding_dim': embedding_dim,
            'train_path': str(output_path / 'train.yaml'), 'eval_path': str(output_path / 'eval.yaml'),
            'dictionary_path': str(output_path / 'dictionary.yaml'),
            'reverse_dictionary_path': str(output_path / 'reverse_dictionary.yaml')}
    if embedding_file is not None:
        info.update({'embedding_path': str(output_path / 'embedding.yaml')})
    yaml_write(output_path / 'info.yaml', info)
    print('导出字典,训练和验证集完成')


if __name__ == '__main__':
    dataset, class_type = read_data(str(dataset_file))
    vocabulary, vocabulary_size = build_vocabulary(dataset)
    build_dictionary(vocabulary, class_type)
    build_dataset(dataset, len(class_type), vocabulary_size)
