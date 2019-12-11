import numpy as np

from scripts.build_en_dataset import read_data
from utils import yaml_utils


class MultipleFileDataGenerator:
    def __init__(self, dictionary, is_augmented, dataset_list, batch_size, seq_length, **kwargs):
        self.word_dictionary = dictionary['word_dictionary']
        self.label_dictionary = dictionary['label_dictionary']
        self.is_augmented = is_augmented
        self.dataset_list = dataset_list
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.kwargs = kwargs

    def get_labels(self):
        return [x for x in self.label_dictionary]

    def get_reverse_dictionary(self):
        reverse_dictionary = self.kwargs['reverse_dictionary']
        self.reverse_dictionary = reverse_dictionary['word_dictionary']
        self.reverse_label_dictionary = reverse_dictionary['label_dictionary']

    def get_size(self):
        return len(self.dataset_list) // self.batch_size

    def get_label(self, label_id):
        return self.reverse_label_dictionary[label_id]

    def get_words(self, word_ids):
        result = ''
        for i in word_ids:
            if i == 0:
                continue
            result += ' ' + self.reverse_dictionary[i]
        return result

    def get_batch_size(self):
        return self.batch_size

    def transform_word(self, source_input_path, source_label):
        source_input_list = read_data(source_input_path)
        source_input = []
        for content in source_input_list:
            source_input.extend(content.split())

        # 固定文本序列长度
        input = [self.word_dictionary[x] for x in source_input if x in self.word_dictionary]
        if len(input) < self.seq_length:
            input.extend([0 for _ in range(self.seq_length - len(input))])
        else:
            input = input[0:self.seq_length]
        # one-hot 处理
        label_id = self.label_dictionary[source_label]
        label = np.zeros((len(self.label_dictionary)), dtype=np.float32)
        label[label_id] = 1.0
        return input, label

    def get_data_generator(self):
        batch_input = list()
        batch_label = list()
        while True:
            if self.is_augmented:
                np.random.shuffle(self.dataset_list)
            for item in self.dataset_list:
                transf_input, transf_label = self.transform_word(item['input'], item['label'])
                batch_input.append(transf_input)
                batch_label.append(transf_label)
                if len(batch_input) == self.batch_size:
                    yield np.array(batch_input), np.array(batch_label)
                    batch_input = list()
                    batch_label = list()


if __name__ == '__main__':
    dataset_info = yaml_utils.read('../dataset/aclImdb/info.yaml')
    dictionary = yaml_utils.read(dataset_info['dictionary_path'])
    train_dataset = yaml_utils.read(dataset_info['eval_path'])
    print('读取完毕')
    data_generator = MultipleFileDataGenerator(dictionary, True, train_dataset, 32, 600)
    batch_input, batch_label = next(data_generator.get_data_generator())
    print(11)
