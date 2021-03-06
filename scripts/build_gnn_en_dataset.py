import csv
import re
import nltk
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from collections import defaultdict
from math import log
from nltk.corpus import stopwords
from pathlib import Path

dataset_name = 'aclImdb'
embedding_dim = 300  # 词向量维度
min_words_freq = 5  # to remove rare words
train_scale = 0.9  # slect 90% training set
window_size = 20  # word co-occurence with context windows
dataset_fold = Path('/home/yf/dataset/{}/train'.format(dataset_name)).absolute()  # E:/Dataset/{}/train
output_dir = Path('../dataset').absolute()
output_path = output_dir / dataset_name
output_path.mkdir(exist_ok=True, parents=True)

global_words_freq = defaultdict(int)  # 统计全部单词评论


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def csv_write(path, list_data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(list_data)


def read_data(path):
    data_path = Path(path)
    contents = list()
    with data_path.open(mode='r', encoding='UTF-8') as file:
        for line in file:
            # 之前发现训练结果不好，然后发现原来时由于英文数据符合断句问题导致的
            while '\n' in line:
                line = line.replace('\n', ' ')
            while '<br />' in line:
                line = line.replace('<br />', ' ')
            line = clean_str(line.strip())
            _words = line.split()
            for _word in _words:
                global_words_freq[_word] += 1
            contents.extend(_words)
    return contents


def remove_words(_dataset):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    vocab = set()
    for i, _item in enumerate(_dataset):
        _words = list()
        for _word in _item['words']:
            if _word not in stop_words and global_words_freq[_word] >= min_words_freq:
                # build vocabulary
                vocab.add(_word)
                _words.append(_word)
        _dataset[i]['words'] = _words
    return _dataset, list(vocab)


def build_adjacency_matrix(_dataset, num_label):
    x_row = list()
    x_col = list()
    x_data = list()
    _data_size = len(_dataset)
    for i in range(_data_size):
        _document = np.array([0.0 for _ in range(embedding_dim)])
        _words = _dataset[i]['words']
        num_word = len(_words)
        for j in range(embedding_dim):
            x_row.append(i)
            x_col.append(j)
            x_data.append(_document[j] / num_word)
    x = sp.csr_matrix((x_data, (x_row, x_col)), shape=(_data_size, embedding_dim))

    y = np.zeros((_data_size, num_label))
    for i in range(_data_size):
        y[i, _dataset[i]['label_id']] = 1
    return x, y, x_row, x_col, x_data


def build_all_adjacency_matrix(_rows, _cols, _data, y, vocab_size):
    _train_size = y.shape[0]
    num_label = y.shape[1]
    word_vectors = np.random.uniform(-0.01, 0.01, (vocab_size, embedding_dim))
    for i in range(vocab_size):
        for j in range(embedding_dim):
            _rows.append(i + _train_size)
            _cols.append(j)
            _data.append(word_vectors.item(i, j))
    all_x = sp.csr_matrix((_data, (_rows, _cols)), shape=(_train_size + vocab_size, embedding_dim))
    all_y = np.concatenate((y, np.zeros((vocab_size, num_label))), 0)
    return all_x, all_y


# name :['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj']
def save_adjacency_matrix(x, name):
    file = open(str(output_path / ('ind.{}.' + name).format(dataset_name)), 'wb')
    pkl.dump(x, file)
    file.close()


def save_dataset_index(x, name):
    csv_write(str(output_path / ('ind.{}.' + name + '.csv').format(dataset_name)), x)


if __name__ == '__main__':
    dataset = list()
    labels = list()
    document = list()
    document_id = 0
    # build clean data
    for _type in dataset_fold.iterdir():
        if _type.is_dir() and _type.name != 'unsup':
            labels.append(_type.name)
            label_id = labels.index(_type.name)
            for item in _type.iterdir():
                document.append(str(item))
                dataset.append({'document_id': document_id, 'label_id': label_id, 'words': read_data(item)})
                document_id += 1
    print('remove_words')
    dataset, vocabulary = remove_words(dataset)
    csv_write(output_path / 'labels.csv', labels)
    print('get dataset labels vocabulary document')
    np.random.shuffle(dataset)
    data_size = len(dataset)

    # x: feature vectors of training docs, no initial features
    print('build train dataset')
    train_size = int(data_size * train_scale)
    train_dataset = dataset[:train_size]
    train_ids = [_item['document_id'] for _item in train_dataset]
    train_x, train_y, rows_, cols_, data_ = build_adjacency_matrix(train_dataset, len(labels))
    save_dataset_index(train_ids, 'train')
    save_adjacency_matrix(train_x, 'x')
    save_adjacency_matrix(train_y, 'y')

    print('build test dataset')
    test_size = data_size - train_size
    test_dataset = dataset[train_size:]
    test_ids = [_item['document_id'] for _item in test_dataset]
    test_x, test_y, _, _, _ = build_adjacency_matrix(test_dataset, len(labels))
    save_dataset_index(test_ids, 'test')
    save_adjacency_matrix(test_x, 'tx')
    save_adjacency_matrix(test_y, 'ty')

    print('build the the feature vectors of both labeled and unlabeled training instances')
    # allx: the the feature vectors of both labeled and unlabeled training instances
    # (a superset of x)
    # unlabeled training instances -> words
    vocabulary_size = len(vocabulary)
    all_x, all_y = build_all_adjacency_matrix(rows_, cols_, data_, train_y, vocabulary_size)
    save_adjacency_matrix(all_x, 'allx')
    save_adjacency_matrix(all_y, 'ally')

    # Doc word heterogeneous graph
    # word co-occurence with context windows
    print('build window')
    windows = list()
    for item_ in dataset:
        words = item_['words']
        num_word = len(words)
        if num_word <= window_size:
            windows.append(words)
        else:
            for j in range(num_word - window_size + 1):
                word_window = words[j: j + window_size]
                windows.append(word_window)

    print('count word in window frequency')
    word_window_freq = defaultdict(int)
    for window in windows:
        appeared = set()
        num_window_ = len(window)
        for i in range(num_window_):
            if window[i] not in appeared:
                word_window_freq[window[i]] += 1
                appeared.add(window[i])

    print('count vocabulary and vocabulary frequency')
    vocabulary_dict = {word_: _id for _id, word_ in enumerate(vocabulary)}
    word_pair_count = defaultdict(int)
    for _, window in enumerate(windows):
        num_window_ = len(window)
        for i in range(1, num_window_):
            for j in range(0, i):
                word_i_id, word_j_id = vocabulary_dict[window[i]], vocabulary_dict[window[j]]
                if word_i_id != word_j_id:
                    # two orders
                    word_pair_count[(word_i_id, word_j_id)] += 1
                    word_pair_count[(word_j_id, word_i_id)] += 1

    print('build adjacency_matrix in vocabulary')
    rows, cols, weight = list(), list(), list()
    num_window = len(windows)
    for (word_i_id, word_j_id), count in word_pair_count.items():
        word_freq_i = word_window_freq[vocabulary[word_i_id]]
        word_freq_j = word_window_freq[vocabulary[word_j_id]]
        # pmi as weights
        pmi = log((1.0 * count / num_window) / (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi > 0:
            rows.append(train_size + word_i_id)
            cols.append(train_size + word_j_id)
            weight.append(pmi)

    # doc word frequency
    print('count document in word frequency')
    doc_word_freq = defaultdict(int)
    for i in range(data_size):
        words = dataset[i]['words']
        for word in words:
            word_id = vocabulary_dict[word]
            doc_word_freq[(i, word_id)] += 1

    print('count word in document frequency')
    word_doc_freq = defaultdict(int)
    for i in range(data_size):
        words = dataset[i]['words']
        appeared = set()
        for word in words:
            if word not in appeared:
                word_doc_freq[word] += 1

    print('build adjacency_matrix in train dataset and test dataset')
    for i in range(data_size):
        words = dataset[i]['words']
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            word_id = vocabulary_dict[word]
            freq = doc_word_freq[(i, word_id)]
            if i < train_size:
                rows.append(i)
            else:
                rows.append(vocabulary_size + i)
            cols.append(train_size + word_id)
            idf = log(1.0 * data_size / word_doc_freq[vocabulary[word_id]])
            weight.append(freq * idf)
            doc_word_set.add(word)
    node_size = train_size + vocabulary_size + test_size
    adjacency_matrix = sp.csr_matrix((weight, (rows, cols)), shape=(node_size, node_size))
    save_adjacency_matrix(adjacency_matrix, 'adj')
