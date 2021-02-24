# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/12/3 7:30 下午
# @Author: wuchenglong

import tensorflow as tf
import pandas as pd
import os
from io import StringIO
import json
from datetime import datetime
from kashgari.tasks.abs_task_model import ABCTaskModel


def load_raw_data(file_path):
    articles = list()  # store training_set [content,content,...]
    labeling_df_list = []

    with open(file_path, 'r', encoding='utf8') as f:
        file_text = f.read().encode('utf-8').decode('utf-8-sig')
    data_set = file_text.split('\n\n--------------------\n\n')[:-1]
    for data in data_set:
        data = data.split('\n')
        content = data[0]
        articles.append(content)
        labeling_data = '\n'.join(data[1:])
        labeling_df_list.append(pd.read_csv(StringIO(labeling_data), sep='\t'))

    labeling_df = pd.concat(labeling_df_list, ignore_index=True)

    return articles, labeling_df


def build_vocab(corpus_file_list, vocab_file, tag_file):
    words = set()
    tags = set()
    for file in corpus_file_list:
        for line in open(file, "r", encoding='utf-8').readlines():
            line = line.strip()
            if line == "end":
                continue
            try:
                w, t = line.split()
                words.add(w)
                tags.add(t)
            except Exception as e:
                print(line.split())
                # raise e

    if not os.path.exists(vocab_file):
        with open(vocab_file, "w", encoding='utf-8') as f:
            for index, word in enumerate(["<UKN>"] + list(words)):
                f.write(word + "\n")

    tag_sort = {
        "O": 0,
        "B": 1,
        "I": 2,
        "E": 3,
    }

    tags = sorted(list(tags),
                  key=lambda x: (len(x.split("-")), x.split("-")[-1], tag_sort.get(x.split("-")[0], 100)))
    if not os.path.exists(tag_file):
        with open(tag_file, "w") as f:
            for index, tag in enumerate(["<UKN>"] + tags):
                f.write(tag + "\n")


def read_vocab(vocab_file):
    vocab2id = {}
    id2vocab = {}
    for index, line in enumerate([line.strip() for line in open(vocab_file, "r", encoding='utf8').readlines()]):
        vocab2id[line] = index
        id2vocab[index] = line
    return vocab2id, id2vocab


def tokenize(files, vocab2id, tag2id):
    contents = []
    labels = []
    content = []
    label = []
    for f in files:
        with open(f, 'r', encoding='utf-8') as fr:
            for line in [elem.strip() for elem in fr.readlines()][:500000]:
                try:
                    if line != "end":
                        w, t = line.split()
                        content.append(vocab2id.get(w, 0))
                        label.append(tag2id.get(t, 0))
                    else:
                        if content and label:
                            contents.append(content)
                            labels.append(label)
                        content = []
                        label = []
                except Exception as e:
                    print(e)
                    content = []
                    label = []

    contents = tf.keras.preprocessing.sequence.pad_sequences(contents, padding='post')
    labels = tf.keras.preprocessing.sequence.pad_sequences(labels, padding='post')
    return contents, labels


def load_test_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        blocks = content.split('\n\n--------------------\n\n')[:-1]
        for block in blocks:
            text = block.split('\n')[1]
            data.append(text)

    return data


tag_check = {
    "I": ["B", "I"],
    "E": ["B", "I"],
}


def check_label(front_label, follow_label):
    if not follow_label:
        raise Exception("follow label should not both None")

    if not front_label:
        return True

    if follow_label.startswith("B-"):
        return False

    if (follow_label.startswith("I-") or follow_label.startswith("E-")) and \
            front_label.endswith(follow_label.split("-")[1]) and \
            front_label.split("-")[0] in tag_check[follow_label.split("-")[0]]:  # I或E的前面要是B或I
        return True
    return False


def format_result(chars, tags):
    entities = []
    entity = []
    for index, (char, tag) in enumerate(zip(chars, tags)):
        entity_continue = check_label(tags[index - 1] if index > 0 else None, tag)
        if not entity_continue and entity:
            entities.append(entity)
            entity = []
        entity.append([index, char, tag, entity_continue])
    if entity:
        entities.append(entity)

    entities_result = []
    for entity in entities:
        if entity[0][2].startswith("B-"):
            entities_result.append(
                {"start_position": entity[0][0],
                 "end_position": entity[-1][0] + 1,
                 "entity_text": "".join([char for _, char, _, _ in entity]),
                 "entity_type": entity[0][2].split("-")[1]
                 }
            )

    return entities_result


def split_chunks(data, chunk_size=511):
    dataset = tf.data.Dataset.from_tensor_slices(data).batch(chunk_size)
    data_chunks = [list(map(lambda x: x.decode(), chunk)) for chunk in list(dataset.as_numpy_iterator())]

    return data_chunks


def save_model(model, history, save_dir):
    date = datetime.now().strftime('%Y%m%d')
    model.save(f'{save_dir}/{date}/')

    history_dir = f'{save_dir}/{date}/history'
    perf = pd.DataFrame(data=history.history)
    if not os.path.exists(history_dir):
        os.mkdir(history_dir)

    perf.to_csv(f'{history_dir}/history_{len(os.listdir(history_dir))}.csv', index=False)


def load_model(model_dir, model_cls: ABCTaskModel):
    model = model_cls.load_model(model_dir)
    info_filepath = os.path.join(model_dir, 'model_info.json')
    if os.path.exists(info_filepath):
        with open(info_filepath, mode='r', encoding='utf-8') as f:
            model_info = json.load(f)
    else:
        model_info = None
    return model, model_info


def save_history(history, save_dir):
    perf = pd.DataFrame(data=history.history)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    perf.to_csv(f'{save_dir}/history_{len(os.listdir(save_dir))}.csv', index=False)
