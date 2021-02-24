import os
import numpy as np

import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.model_selection import train_test_split


def loadInputFile(path):
    training_set = list()  # store training_set [content,content,...]
    position = list()  # store position [article_id, start_pos, end_pos, entity_text, entity_type, ...]
    mentions = dict()  # store mentions[mention] = Type
    with open(path, 'r', encoding='utf8') as f:
        file_text = f.read().encode('utf-8').decode('utf-8-sig')
    data_set = file_text.split('\n\n--------------------\n\n')[:-1]
    for data in data_set:
        data = data.split('\n')
        content = data[0]
        training_set.append(content)
        annotations = data[1:]
        for annot in annotations[1:]:
            annot = annot.split('\t')
            position.extend(annot)
            mentions[annot[3]] = annot[4]
    
    return training_set, position, mentions


def CRFFormatData(training_set, position, path):
    if os.path.isfile(path):
        os.remove(path)
    output_file = open(path, 'a', encoding='utf-8')

    # output file lines
    count = 0  # annotation counts in each content
    tagged = list()
    for article_id in range(len(training_set)):
        training_set_split = list(training_set[article_id])
        while '' or ' ' in training_set_split:
            if '' in training_set_split:
                training_set_split.remove('')
            else:
                training_set_split.remove(' ')
        start_tmp = 0
        for position_idx in range(0, len(position), 5):
            if int(position[position_idx]) == article_id:
                count += 1
                if count == 1:
                    start_pos = int(position[position_idx+1])
                    end_pos = int(position[position_idx+2])
                    entity_type = position[position_idx+4]
                    if start_pos == 0:
                        token = list(training_set[article_id][start_pos:end_pos])
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ', '')) == 0:
                                continue
                            # BIO states
                            if token_idx == 0:
                                label = 'B-'+entity_type
                            else:
                                label = 'I-'+entity_type
                            
                            output_str = token[token_idx] + ' ' + label + '\n'
                            output_file.write(output_str)

                    else:
                        token = list(training_set[article_id][0:start_pos])
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ', '')) == 0:
                                continue
                            
                            output_str = token[token_idx] + ' ' + 'O' + '\n'
                            output_file.write(output_str)

                        token = list(training_set[article_id][start_pos:end_pos])
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ', '')) == 0:
                                continue
                            # BIO states
                            if token[0] == '':
                                if token_idx == 1:
                                    label = 'B-'+entity_type
                                else:
                                    label = 'I-'+entity_type
                            else:
                                if token_idx == 0:
                                    label = 'B-'+entity_type
                                else:
                                    label = 'I-'+entity_type

                            output_str = token[token_idx] + ' ' + label + '\n'
                            output_file.write(output_str)

                    start_tmp = end_pos
                else:
                    start_pos = int(position[position_idx+1])
                    end_pos = int(position[position_idx+2])
                    entity_type = position[position_idx+4]
                    if start_pos < start_tmp:
                        continue
                    else:
                        token = list(training_set[article_id][start_tmp:start_pos])
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ', '')) == 0:
                                continue
                            output_str = token[token_idx] + ' ' + 'O' + '\n'
                            output_file.write(output_str)

                    token = list(training_set[article_id][start_pos:end_pos])
                    for token_idx in range(len(token)):
                        if len(token[token_idx].replace(' ', '')) == 0:
                            continue
                        # BIO states
                        if token[0] == '':
                            if token_idx == 1:
                                label = 'B-'+entity_type
                            else:
                                label = 'I-'+entity_type
                        else:
                            if token_idx == 0:
                                label = 'B-'+entity_type
                            else:
                                label = 'I-'+entity_type
                        
                        output_str = token[token_idx] + ' ' + label + '\n'
                        output_file.write(output_str)
                    start_tmp = end_pos

        token = list(training_set[article_id][start_tmp:])
        for token_idx in range(len(token)):
            if len(token[token_idx].replace(' ', '')) == 0:
                continue
            
            output_str = token[token_idx] + ' ' + 'O' + '\n'
            output_file.write(output_str)

        count = 0
    
        output_str = 'end\n'
        output_file.write(output_str)

        if article_id % 10 == 0:
            print('Total complete articles:', article_id)

    # close output file
    output_file.close()


def CRF(x_train, y_train, x_test, y_test):
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(x_train, y_train)

    y_pred = crf.predict(x_test)
    y_pred_mar = crf.predict_marginals(x_test)

    labels = list(crf.classes_)
    labels.remove('O')
    f1score = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))  # group B and I results
    print(flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
    return y_pred, y_pred_mar, f1score


def Dataset(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    data_list, data_list_tmp = list(), list()
    article_id_list = list()
    idx = 0
    for row in data:
        if row == '\n':
            article_id_list.append(idx)
            idx += 1
            data_list.append(data_list_tmp)
            data_list_tmp = []
        else:
            row = row.strip('\n').split(' ')
            data_tuple = (row[0], row[1])
            data_list_tmp.append(data_tuple)
    if len(data_list_tmp) != 0:
        data_list.append(data_list_tmp)
    
    # here we random split data into training dataset and testing dataset
    # but you should take `development data` or `test data` as testing data
    # At that time, you could just delete this line, 
    # and generate data_list of `train data` and data_list of `development/test data` by this function
    traindata_list, testdata_list, traindata_article_id_list, testdata_article_id_list = \
        train_test_split(data_list, article_id_list, test_size=0.33, random_state=42)
    
    return data_list, traindata_list, testdata_list, traindata_article_id_list, testdata_article_id_list 


def Word2Vector(data_list, embedding_dict):
    embedding_list = list()

    # No Match Word (unknown word) Vector in Embedding
    unk_vector = np.random.rand(*list(embedding_dict.values())[0].shape)

    for idx_list in range(len(data_list)):
        embedding_list_tmp = list()
        for idx_tuple in range(len(data_list[idx_list])):
            key = data_list[idx_list][idx_tuple][0]  # token

            if key in embedding_dict:
                value = embedding_dict[key]
            else:
                value = unk_vector
            embedding_list_tmp.append(value)
        embedding_list.append(embedding_list_tmp)
    return embedding_list


def Feature(embed_list):
    feature_list = list()
    for idx_list in range(len(embed_list)):
        feature_list_tmp = list()
        for idx_tuple in range(len(embed_list[idx_list])):
            feature_dict = dict()
            for idx_vec in range(len(embed_list[idx_list][idx_tuple])):
                feature_dict['dim_' + str(idx_vec+1)] = embed_list[idx_list][idx_tuple][idx_vec]
            feature_list_tmp.append(feature_dict)
        feature_list.append(feature_list_tmp)
    return feature_list


def Preprocess(data_list):
    label_list = list()
    for idx_list in range(len(data_list)):
        label_list_tmp = list()
        for idx_tuple in range(len(data_list[idx_list])):
            label_list_tmp.append(data_list[idx_list][idx_tuple][1])
        label_list.append(label_list_tmp)
    return label_list
