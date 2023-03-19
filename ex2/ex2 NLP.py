from nltk.corpus import brown
from itertools import chain


##############################################################
# Create test and train sets: a
##############################################################

def extract_tag_prefix(dataset):
    """
    extract tags prefix from a given dataset
    :param dataset: dataset to extract tags prefix from
    :return: the new data
    """
    new_dataset = []
    for sentence in dataset:
        new_sentence = []
        for word, tag in sentence:
            tag = tag.split('-')[0]
            tag = tag.split('+')[0]
            tag = tag.split('*')[0]
            new_sentence.append((word, tag))
        new_dataset.append(new_sentence)
    return new_dataset


def create_news_test_and_train():
    """
    load the corpus and divide it to test and set
    :return: train and test sets
    """
    news = list(brown.tagged_sents(categories='news'))
    news = extract_tag_prefix(news)
    div_param = int(len(news) / 10 * 9)
    train = news[:div_param]
    test = news[div_param:]
    return train, test


##############################################################
# Most likely tag: b-i, b-ii
##############################################################

def fit_MLT(train):
    """
    fit a most likely tag model
    :param train: train test
    :return: dictionary of words and category per word with count
    """
    words_dict = dict()
    for sentence in train:
        for word, tag in sentence:
            if word not in words_dict:
                words_dict[word] = {tag: 1}
            else:
                word_tags_dict = words_dict[word]
                if tag not in word_tags_dict:
                    word_tags_dict[tag] = 0
                word_tags_dict[tag] += 1
    return words_dict


def predict_MLT(train_dict, test):
    """
    predict the Most likely tag
    :param train_dict: dictionary of words and category per word with count
    :param test: test set
    :return: prediction of tag for every word, and 2 lists of all known
            and unknown words
    """
    known_words_, unknown_words_ = [], []
    predict_dict = dict()
    for tuple_ in test:
        word = tuple_[0]
        if word in train_dict:
            predict_dict[word] = max(train_dict[word],
                                     key=train_dict[word].get)
            known_words_.append(tuple_)
        else:
            predict_dict[word] = "NN"
            unknown_words_.append(tuple_)
    return predict_dict, known_words_, unknown_words_


def accuracy(predict_dict, test):
    """
    calculate the model accuracy
    :param predict_dict: prediction of tag for every word
    :param test: test set
    :return: the model accuracy
    """
    correct_guesses = 0
    for word, tag in test:
        if predict_dict[word] == tag:
            correct_guesses += 1
    return correct_guesses / len(test)


def most_likely_tag(train, test):
    """
    The main function for questions - a,b
    :param train: train set
    :param test: test set
    :return predict tag for every word in test set
    """
    test = list(chain.from_iterable(test))
    train_words_dict_ = fit_MLT(train)
    predict_dict_, known_words_, unknown_words_ = predict_MLT(
        train_words_dict_,
        test)

    known_words_accuracy = accuracy(predict_dict_, known_words_)
    print("Known words Error is: " + str(1 - known_words_accuracy))

    unknown_words_accuracy = accuracy(predict_dict_, unknown_words_)
    print("Unknown words Error is: " + str(1 - unknown_words_accuracy))

    total_accuracy = accuracy(predict_dict_, test)
    print("Total Error is: " + str(1 - total_accuracy))

    return predict_dict_, unknown_words_, train_words_dict_


##############################################################
# Bigram HMM tagger: c-i, c-ii, c-iii
##############################################################

def train_bigram_HMM(train_set):
    """
    Train a HMM Bigram model
    :param train_set: the train set
    :return: two dictionaries when -
    next_category_dict: key = category, val = another dict - next categories
                for every next category with counter.
    words_in_category_dict: key = category, val = another dict - category words
                for every word in category with counter.
    """
    next_category_dict, words_in_category_dict = dict(), dict()
    for sentence in train_set:
        sentence.append(('STOP', 'STOP'))  # add STOP
        prev_category = "*"
        for word, tag in sentence:
            if prev_category not in next_category_dict.keys():
                next_category_dict[prev_category] = {tag: 1}
            else:  # word already in dict
                next_categories = next_category_dict[prev_category]
                if tag not in next_categories.keys():
                    next_categories[tag] = 0
                next_categories[tag] += 1
            if tag not in words_in_category_dict.keys():
                words_in_category_dict[tag] = {word: 1}
            else:
                words_in_category = words_in_category_dict[tag]
                if word not in words_in_category_dict[tag].keys():
                    words_in_category[word] = 0
                words_in_category[word] += 1
            prev_category = tag
    return next_category_dict, words_in_category_dict


def viterbi(sentence, next_category_dict, words_in_category_dict):
    """
    Implement Viterbi algorithm
    :param sentence: a sentence to tag
    :param next_category_dict: a dict with - key = category,
    val = another dict - next categories for every next category with counter.
    :param words_in_category_dict: a dict with - key = category,
    val = another dict - category words for every word in category with counter
    :return: the tag sequence
    """
    return_sequence = []
    # for word in sentence:
    return return_sequence


##############################################################
# Add-one smoothing: d
##############################################################

def add_one_smoothing(dataset, test_set):
    """

    :param test_set:
    :param dataset:
    :return:
    """
    next_category_dict, words_in_category_dict = train_bigram_HMM(dataset)
    delta_q_dict, delta_e_dict, delta_c, delta_w = \
        train_smoothing(dataset, next_category_dict, words_in_category_dict)

    test_prediction_ = []
    for sentence in test_set:
        sentence.append(("STOP", "STOP"))  # todo is "stop" needed?
        tag_sequence_ = viterbi(sentence, delta_q_dict, delta_e_dict)
        test_prediction_.append(tag_sequence_)

    error_rate = smoothing_error(test_prediction_, test_set, delta_c, delta_w)


def train_smoothing(dataset, next_category_dict, words_in_category_dict):
    """

    :param dataset:
    :param next_category_dict:
    :param words_in_category_dict:
    :return:
    """
    # find all categories and all words
    all_categories, all_words = [], []
    for sentence in dataset:
        for word, tag in sentence:
            if tag not in all_categories:
                all_categories.append(tag)
            if word not in all_words:
                all_words.append(word)

    # add one for every next category
    delta_c = 0
    for category in next_category_dict.keys():
        for cat_to_add in all_categories:
            next_categories = next_category_dict[category]
            if cat_to_add not in next_categories:
                next_categories[cat_to_add] = 0
            next_categories[cat_to_add] += 1
            delta_c += 1

    # add one for every word in any category
    delta_w = 0
    for category in words_in_category_dict.keys():
        for word_to_add in all_words:
            category_words = words_in_category_dict[category]
            if word_to_add not in category_words:
                category_words[word_to_add] = 0
            category_words[word_to_add] += 1
            delta_w += 1

    return next_category_dict, words_in_category_dict, delta_c, delta_w


def smoothing_error(test_prediction_, test_set, delta_c, delta_w):
    """

    :param test_prediction_:
    :param test_set:
    :param delta_c:
    :param delta_w:
    :return:
    """
    correct_guesses = 0
    test_len = len(list(chain.from_iterable(test_set)))
    for sent in test_set:
        for word, tag in sent:
            # if test_prediction_[word] == tag:  # todo check if equal is correct
            correct_guesses += 1
    return correct_guesses / (test_len + delta_w)  # todo what about delta


##############################################################
# Pseudo-words: e
##############################################################

def pseudo_words(unknown_words_, train_words_dict_, train_len):
    create_pseudo_words(unknown_words_, train_words_dict_, train_len)
    pass


def create_pseudo_words(unknown_words_, train_words_dict_, train_len):
    rare_words = []
    # expectation = train_len / len(train_words_dict_.keys())
    for word in train_words_dict_.keys():
        if sum(train_words_dict_[word].values()) < 5:
            rare_words.append({word: train_words_dict_[word]})
    # todo - pseudo words


if __name__ == '__main__':
    # question a: create train and test sets
    news_train, news_test = create_news_test_and_train()

    # question b: find the most likely tag and print Error rate
    predict_tag_d, unknown_words, train_words_dict = \
        most_likely_tag(news_train, news_test)

    # question c_i: train Bigram HMM
    q_dict, e_dict = train_bigram_HMM(news_train)

    # question c_ii: viterbi algorithm
    test_prediction = []
    for sentence_ in news_test:
        sentence_.append(("STOP", "STOP"))  # todo is "stop" needed?
        tag_sequence = viterbi(sentence_, q_dict, e_dict)
        test_prediction.append(tag_sequence)

    # question d: add one smoothing
    add_one_smoothing(news_train + news_test, news_test)

    # question e: pseudo words
    pseudo_words(unknown_words, train_words_dict,
                 len(list(chain.from_iterable(news_train))))


# import nltk
#
# nltk.download('brown')
# from nltk.corpus import brown
# from collections import defaultdict
# import re
# import numpy as np
# import pandas as pd
#
# INITIAL_TAG = '*'
# UNKONWN_TAG = 'NN'
# STOP_TAG = 'STOP'
# THRESHOLD = 5
#
#
# def prefix(tag):
#     return tag.split('+')[0].split('-')[0].split('*')[0]
#
# def create_news_test_and_train():
#     """
#     load the corpus and divide it to test and set
#     :return: train and test sets
#     """
#     news = brown.tagged_sents(categories='news')
#     div_param = int(len(news) * (9 / 10))
#     train = news[:div_param]
#     test = news[div_param:]
#     return train, test
#
#
# train_set, test_set = create_news_test_and_train()
#
# ###############################
# # MLE b
# ###############################
# def train_model(training_sent):
#     count = defaultdict(lambda: defaultdict(int))
#     for sent in training_sent:
#         for word in sent:
#             count[word[0]][prefix(word[1])] += 1
#     return count
#
#
# def get_error(test_set, count):
#     total, in_N, N, in_c = 0, 0, 0, 0
#     for sent in test_set:
#         for word in sent:
#             N += 1
#             he = word[0]
#             p = (max(count[he]) if he in count else 'NN') == prefix(word[1])
#             total += p
#             if word[0] in count:
#                 in_N += 1
#                 in_c += p
#     total_acc = total / N
#     in_acc = in_c / in_N
#     out_acc = (total - in_c) / (N - in_N)
#     return 1 - in_acc, 1 - out_acc, 1 - total_acc
#
#
# mle = train_model(train_set)
# for_b = get_error(test_set, mle)
# print(for_b)
#
# ###############################
# # 3 c,d,e
# ###############################
#
# # helper func for the next part
# PSEUDOWORDS = {
#     "\d+.{0,1}\d*$": 'NUM',
#     "-year-old$": 'AGE',
#     "[$]": 'PRICE',
#     "^\d+/\d+/{0,1}\d*$": 'DATE',
#     "^\d+-\d+-{0,1}\d*$": 'digitsAndDash',
#     "^[A-Z]+$": 'ALLCAPS',
#     "^[A-Za-z][.][A-Za-z]([.][A-Za-z])*$": 'INITIALS',
#     "\w+\\.": 'capPeriod',
#     "[A-Z]\w+": 'initCap'
# }
#
#
# def pseudo_word(xi):
#     for pat in PSEUDOWORDS.keys():
#         if re.findall(pat, xi, re.I):
#             return PSEUDOWORDS[pat]
#     return 'otherTag'
#
#
# def tag_prefix(tag):
#     return tag.split('+')[0].split('-')[0].split('*')[0]
#
#
# ############################################
#
# def init_tag_word(p_word, train_set):
#     """
#     :param p_word: if using in pseudo word
#     :param train_set:
#     :return:
#     """
#     tags_data = defaultdict(lambda: [defaultdict(int), defaultdict(int)])
#     indexed_tags = {}
#     for j in range(len(train_set)):
#         s = train_set[j]
#         for word, tag in s:
#             tags_data[tag_prefix(tag)][0][word] += 1
#             if tag_prefix(tag) not in indexed_tags:
#                 indexed_tags[tag_prefix(tag)] = len(indexed_tags.keys())
#
#     if p_word:
#         for tag in tags_data.keys():
#             words = tags_data[tag][0].copy()
#             for word in words:
#                 if tags_data[tag][0][word] < THRESHOLD:
#                     tags_data[tag][0][pseudo_word(word)] += \
#                         tags_data[tag][0].pop(word)
#     for j in range(len(train_set)):
#         s = train_set[j]
#         former_tag = INITIAL_TAG
#         for word, tag in s + [('', STOP_TAG)]:
#             tag = tag.split('+')[0].split('-')[0].split('*')[0]
#             tags_data[former_tag][1][tag] += 1
#             former_tag = tag
#     return tags_data, indexed_tags
#
#
# ############################################
#
# def viterbi_build_table(sentence, emission_prob, transition_prob, smoot,
#                         tags_data, indexed_tags):
#     tags = list(tags_data.keys())
#
#     n = len(sentence)
#     viterbi_table = [{tag: (1, None) for tag in tags} for i in range(n)]
#
#     for k in range(n):
#         for tag_i in tags:
#             prev_tag, max_prob = None, 0
#             e = emission_prob(sentence[k], tag_i, smoot, tags_data)
#
#             if k == 0:
#                 viterbi_table[k][tag_i] = (
#                 e * transition_prob(tag_i, INITIAL_TAG, tags_data),
#                 INITIAL_TAG)
#                 continue
#
#             if e != 0:
#                 for tag_j in tags:
#                     pi = viterbi_table[k - 1][tag_j][0]
#                     if pi == 0:
#                         continue
#                     q = transition_prob(tag_i, tag_j, tags_data)
#
#                     if pi * q > max_prob:
#                         max_prob = pi * q
#                         prev_tag = tag_j
#
#             if prev_tag is None:
#                 prev_tag = UNKONWN_TAG
#
#             viterbi_table[k][tag_i] = (max_prob * e, prev_tag)
#
#     return viterbi_table
#
#
# def viterbi_infrence(sentence, emission_prob, transition_prob, smoot,
#                      tags_data, indexed_tags):
#     n = len(sentence)
#     viterbi_table = viterbi_build_table(sentence, emission_prob,
#                                         transition_prob, smoot, tags_data,
#                                         indexed_tags)
#     return_val = []
#     y = max(viterbi_table[n - 1], key=lambda key: viterbi_table[n - 1][key])
#
#     for k in range(n - 1, -1, -1):
#         return_val.append(y)
#         y = viterbi_table[k][y][1]
#     return_val.reverse()
#
#     return return_val
#
#
# def transition_prob(yj, yi, tags_data):
#     return tags_data[yi][1][yj] / sum(tags_data[yi][1].values())

#
# def get_error(test_set, emission_prob, transition_prob, smoot, tags_data,
#               indexed_tags):
#     good_uk_guesses, bad_uk_guesses, good_guesses, bad_guesses, = 0, 0, 0, 0
#     for j in range(len(test_set)):
#         sent = [tup[0] for tup in test_set[j]]
#         viterbi_tags = viterbi_infrence(sent, emission_prob, transition_prob,
#                                         smoot, tags_data, indexed_tags)
#         for i in range(len(sent)):
#             if viterbi_tags[i] == test_set[j][i][1]:
#                 if viterbi_tags[i] != UNKONWN_TAG:
#                     good_guesses += 1
#                 else:
#                     good_uk_guesses += 1
#             else:
#                 if viterbi_tags[i] != UNKONWN_TAG:
#                     bad_guesses += 1
#                 else:
#                     bad_uk_guesses += 1
#
#     err_rate = 1 - (good_guesses / (good_guesses + bad_guesses))
#     uk_err_rate = 1 - (good_uk_guesses / (
#                 good_uk_guesses + bad_uk_guesses) if good_uk_guesses > 0 else 0)
#     total_err = 1 - (
#             (good_guesses + good_uk_guesses) / (
#                 good_guesses + bad_guesses + good_uk_guesses + bad_uk_guesses))
#
#     return err_rate, uk_err_rate, total_err
#
#
# ################################
# # HMM c
# ################################
#
# def emission_prob_c(xi, yi, smoo, tags_data):
#     return tags_data[yi][0][xi] / sum(tags_data[yi][0].values())
#
#
# # hmm_c = init_tag_word(False,train_set)
# # for_c = get_error(test_set,emission_prob_c,transition_prob,False,hmm_c[0],hmm_c[1])
# # print(for_c)
#
#
# ################################
# # HMM d
# ################################
#
# def emission_prob_d(xi, yi, smoo, tags_data):
#     return (tags_data[yi][0][xi] + 1) / (
#                 sum(tags_data[yi][0].values()) + len(tags_data[yi][0]))
#
#
# # hmm_d = init_tag_word(False,train_set)
# # for_d = get_error(test_set,emission_prob_d,transition_prob,False,hmm_d[0],hmm_d[1])
# # print(for_d,'HMM whit smoot')
#
# ################################
# # HMM e
# ################################
#
#
# def emission_prob_e(xi, yi, smoo, tags_data):
#     if tags_data[yi][0][xi] == 0:  #
#         xi = pseudo_word(xi)
#     return emission_prob_d(xi, yi, smoo,
#                            tags_data) if smoo else emission_prob_c(xi, yi,
#                                                                    smoo,
#                                                                    tags_data)
#
#
# hmm_e = init_tag_word(True, train_set)
#
#
# # for_e_no_smoot = get_error(test_set, emission_prob_e, transition_prob, False, hmm_e[0], hmm_e[1])
# # for_e_smoot = get_error(test_set, emission_prob_e, transition_prob, True, hmm_e[0], hmm_e[1])
# # print(for_e_no_smoot)
# # print(for_e_smoot)
#
# ################################
# # confusion matrix (HMM e.iii)
# ################################
#
#
# def confusion_matrix(test_set, indexed_tags):
#     # Iterate through test set to add real tags
#     for i in range(len(test_set)):
#         for tup in test_set[i]:
#             tag = tag_prefix(tup[1])
#             if tag not in indexed_tags:
#                 indexed_tags[tag] = len(indexed_tags.keys())
#     k = len(indexed_tags.keys())
#     conf_mat = np.zeros((k, k))
#
#     for i in range(len(test_set)):
#         # viterbi with pseudo-words and smoothing
#         predictions = viterbi_infrence([tup[0] for tup in test_set[i]],
#                                        emission_prob_e, transition_prob, True,
#                                        hmm_e[0], hmm_e[1])
#         for j in range(len(predictions)):
#             true_index = indexed_tags[tag_prefix(test_set[i][j][1])]
#             predicted_index = indexed_tags[predictions[j]]
#             conf_mat[true_index, predicted_index] += 1
#
#     df = pd.DataFrame(conf_mat, columns=indexed_tags.keys(),
#                       index=indexed_tags.keys())
#     df.to_csv('./confusion matrix.csv')
#     return df
#
#
# confusion_matrix(test_set, init_tag_word(True, train_set)[1])
