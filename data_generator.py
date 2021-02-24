from typing import List, AnyStr

import os
import numpy as np

from baseline import loadInputFile, CRFFormatData


def crf_format_data(rawdata_path, output_path):
    training_set, position, mentions = loadInputFile(rawdata_path)
    CRFFormatData(training_set, position, path=output_path)


def read_data(files, end_flag: str):
    dataset = []
    for fp in files:
        with open(fp, 'r', encoding='utf-8') as f:
            article = []
            for line in f.readlines():
                if line != f'{end_flag}\n':
                    wl = line.strip().split()
                    article.append((wl[0], wl[1]))
                else:
                    dataset.append(article)
                    article = []
    return dataset


def generate_dataset(dataset: List,
                     saving_dir: AnyStr,
                     train_file_name: AnyStr = 'TRAIN',
                     test_file_name: AnyStr = 'TEST',
                     train_ratio: float = 0.9,
                     test_ratio: float = 0.1):
    np.random.shuffle(dataset)
    train_size = int(len(dataset) * train_ratio)
    test_size = int(len(dataset) * test_ratio) + 1

    with open(os.path.join(saving_dir, train_file_name), 'w', encoding='utf-8') as f:
        for article in dataset[:train_size]:
            f.writelines([(' '.join(wl) + '\n') for wl in article])
            f.write('\n')
    with open(os.path.join(saving_dir, test_file_name), 'w', encoding='utf-8') as f:
        for article in dataset[train_size:train_size+test_size]:
            f.writelines([(' '.join(wl) + '\n') for wl in article])
            f.write('\n')
    if train_size + test_size < len(dataset):
        with open(os.path.join(saving_dir, 'dev.txt'), 'w', encoding='utf-8') as f:
            for article in dataset[train_size+test_size:]:
                f.writelines([(' '.join(wl) + '\n') for wl in article])
                f.write('\n')
