from nltk.corpus import dependency_treebank
import nltk
import numpy as np
from Chu_Liu_Edmonds_algorithm import min_spanning_arborescence_nx
from itertools import combinations
from collections import namedtuple, Counter

Arc = namedtuple('Arc', 'head tail weight')


class MST:

    def __init__(self, N_iterations=2, lr=1, is_bonus=False):
        self.train, self.test = self.create_test_and_train_sets()
        self.words_lst, self.tags_lst = self.create_word_tag_sets(
            self.train + self.test)
        self.is_bonus = is_bonus
        self.N_iterations = N_iterations
        self.lr = lr
        self.theta = Counter()

    def create_test_and_train_sets(self):
        """
        load the corpus and divide it to test and set
        :return: train and test sets
        """
        nltk.download('dependency_treebank')
        sentences = dependency_treebank.parsed_sents()
        div_param = int(len(sentences) / 10 * 9)
        train = sentences[:div_param]
        test = sentences[div_param:]
        return train, test

    def create_word_tag_sets(self, dataset):
        """

        :param dataset:
        :return:
        """
        words_set, tags_set = set(), set()
        for sent in dataset:
            for i in range(len(sent.nodes)):
                words_set.add(sent.nodes[i]['word'])
                tags_set.add(sent.nodes[i]['tag'])
        return list(words_set), list(tags_set)

    def feature_function(self, word1_ind, word2_ind, sent):
        """
        create a feature function for 2 given words
        :param word1_ind: index of first word
        :param word2_ind: index of second word
        :param sentence: sentece to find a words in
        :return: feature function, tuple of indexes that represent the words and tags
        """
        # find words and tags
        word1 = sent.nodes[word1_ind]['word']
        word2 = sent.nodes[word2_ind]['word']
        tag1 = sent.nodes[word1_ind]['tag']
        tag2 = sent.nodes[word2_ind]['tag']

        # find words and tags indexes, in words list and tags list.
        word1_lst_ind = self.words_lst.index(word1)
        word2_lst_ind = self.words_lst.index(word2)
        tag1_lst_ind = self.tags_lst.index(tag1)
        tag2_lst_ind = self.tags_lst.index(tag2)

        return (word1_lst_ind, word2_lst_ind), (tag1_lst_ind, tag2_lst_ind)

    def all_possible_arcs(self, len_sent):
        """
        find all posible edges in given sentense
        :param len_sent: sentence's length
        :return: all posible edges in sentense
        """
        return sorted(list(combinations(np.arange(len_sent), 2)) + list(
            combinations(np.flip(np.arange(1, len_sent)), 2)))

    def build_gold_standart_tree(self, sent):
        """
        build the gold standart tree
        :param sent: a sentence to build a tree for
        :return: gold standart tree
        """
        root = sent.root["address"]
        edges = sent.nx_graph().edges
        gold_tree = [(edge[1], edge[0]) for edge in edges] + [(0, root)]
        return gold_tree

    def calculate_edge_weigth(self, w1_ind, w2_ind, sent):
        """
        calculate the wigth for a given edge
        :param w1_ind: index of first word
        :param w2_ind: index of second word
        :param sent: a sentece
        :return: the edge wigth
        """
        words_tuple, tags_tuple = self.feature_function(w1_ind, w2_ind, sent)
        w = self.theta[words_tuple] + self.theta[tags_tuple]
        dist = abs(w1_ind - w2_ind)
        return w if not self.is_bonus else w + (4 / dist)

    def build_mst(self, sent):
        """
         build the maximum spaning tree
        :param sent: a sentence to build a tree for
        :return: maximum spaning tree
        """
        arcs = self.all_possible_arcs(len(sent.nodes))
        arcs_w = []
        for (i, j) in arcs:
            w = self.calculate_edge_weigth(i, j, sent) * -1
            arcs_w.append(Arc(i, j, w))
        mst = min_spanning_arborescence_nx(arcs_w, None)
        return [(mst[k].head, mst[k].tail) for k in mst.keys()]

    def features_sum(self, sent, arcs):
        """
        calculate the sum of every feature in given arcs
        :param sent: a sentence
        :param arcs: all arcs of tree
        :return: the sum of every feature in given arcs
        """
        return sum([Counter(self.feature_function(arc[0], arc[1], sent))
                    for arc in arcs], Counter())

    def perceptron(self, dataset):
        """
        perceptron algorithm, update theta perameter for learning
        :param dataset:
        """
        current_theta = Counter()
        for iter in range(self.N_iterations):
            for sent in dataset:  # todo train or test
                gold_tree = self.build_gold_standart_tree(sent)
                T_tag = self.build_mst(sent)
                gold_sum = self.features_sum(sent, gold_tree)
                T_tag_sum = self.features_sum(sent, T_tag)
                gold_sum.subtract(T_tag_sum)
                current_theta.update(gold_sum)
                current_theta = Counter(
                    {k: v * self.lr for k, v in current_theta.items()})
                self.theta.update(current_theta)
        self.theta = Counter(
            {k: v / (self.N_iterations * len(dataset)) for k, v in
             self.theta.items()})

    def calculate_accuracy(self, sent):
        """
        calculate accuracy of given sentence
        :param sent: a sentence
        :return: accuracy of sentence
        """
        mst = self.build_mst(sent)
        gst = self.build_gold_standart_tree(sent)
        return len(set(mst).intersection(set(gst))) / (len(sent.nodes) - 1)

    def evaluate(self):
        """
        evaluate the accuracy of the computed weights over the test set
        :return: total accuracy over test set
        """
        total_acc = 0
        for sent in self.test:
            total_acc += self.calculate_accuracy(sent)
        return total_acc / len(self.test)


def main():
    """
    cteate an MST object, train a model, and test it.
    print tha accuracy rate over the test set
    """
    mst = MST()
    mst.perceptron(mst.train)
    accuracy = mst.evaluate()
    print("Test evaluation is: " + str(accuracy))

    # with bonus
    mst = MST(is_bonus=True)
    mst.perceptron(mst.train)
    accuracy = mst.evaluate()
    print("Test evaluation for bonus is: " + str(accuracy))


if __name__ == '__main__':
    main()
