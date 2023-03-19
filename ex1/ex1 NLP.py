import math

import spacy
from datasets import load_dataset
import numpy as np


# class for a new word
class Word:
    def __init__(self, w):
        self.w = w
        self.next_words = dict()
        self.count = 0

    def add(self, other):
        if other not in self.next_words:
            self.next_words[other] = 0
        self.next_words[other] += 1

    def __str__(self):
        return self.w + " count is: " + str(self.next_words)

    def __repr__(self):
        return self.w + " count is: " + str(self.next_words)


def parser():
    # load dataset
    nlp = spacy.load("en_core_web_sm")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    start_word = Word('start')
    words_dict = {'start': start_word}

    # parser
    for text in dataset['text']:
        doc = nlp(text)
        prev_word = start_word
        if len(text) > 1:
            start_word.count += 1
        for token in doc:
            if token.is_alpha:
                token = token.lemma_
                if token not in words_dict:
                    words_dict[token] = Word(token)
                words_dict[token].count += 1
                prev_word.add(token)
                prev_word = words_dict[token]
    return words_dict


def num_of_words(words_dict):
    sum_ = 0
    for word in words_dict.values():
        sum_ += word.count
    return sum_


def unigram_prob_f(new_sentence, words_dict):
    nlp = spacy.load("en_core_web_sm")
    new_sentence = nlp(new_sentence)
    all_words = num_of_words(words_dict)
    prob = 0
    prob_per_word = []
    for word_ in new_sentence:
        word_ = word_.lemma_
        word_prob = 0
        if word_ in words_dict:
            word_prob = words_dict[word_].count / all_words
        prob += np.log(word_prob)
        prob_per_word.append(word_prob)
    return prob, prob_per_word


def bigram_prob_f(new_sentence, words_dict):
    nlp = spacy.load("en_core_web_sm")
    new_sentence = nlp(new_sentence)
    first_word = new_sentence[0].lemma_
    prob_per_word = []

    # calculate the total count of the first word in the corpus
    first_word_amount = 0
    if first_word in words_dict['start'].next_words:
        first_word_amount = words_dict['start'].next_words[first_word]
    prob = np.log(first_word_amount / words_dict['start'].count)
    prob_per_word.append(first_word_amount / words_dict['start'].count)
    prev_word = words_dict[first_word]
    for word_ in new_sentence[1:]:
        word_ = word_.lemma_
        word_prob = 0
        if word_ in prev_word.next_words and prev_word.count != 0:
            word_prob = prev_word.next_words[word_] / sum(
                prev_word.next_words.values())
        prob += np.log(word_prob)
        prob_per_word.append(word_prob)
        prev_word = words_dict[word_]
    return prob, prob_per_word


def complete_the_sentence(sentence, words_dict):
    sentence = sentence.split(' ')
    last_word = sentence[-1]
    word_class = words_dict[last_word]
    return max(word_class.next_words, key=word_class.next_words.get)


def bigram_probability_perplexity(sentences, words_dict):
    # calculate probability
    probability_lst = []
    for sentence in sentences:
        probability_lst.append(bigram_prob_f(sentence, words_dict)[0])

    # calculate perplexity
    M = len(sentences[0].split(' ')) + len(sentences[1].split(' '))
    l = sum(probability_lst)
    l /= M
    perplexity = np.power(math.e, -l)

    return probability_lst, perplexity


def linear_interpolation_smoothing(sentences, words_dict):
    probability_lst = []
    for sentence in sentences:
        B_lambda, U_lambda = 2 / 3, 1 / 3
        unigram_prob, u_prob_per_word = unigram_prob_f(sentence, words_dict)
        bigram_prob, b_prob_per_word = bigram_prob_f(sentence, words_dict)
        probability = 0
        for j in range(len(u_prob_per_word)):
            probability += np.log(U_lambda * u_prob_per_word[j] + B_lambda *
                                  b_prob_per_word[j])
        probability_lst.append(probability)

    # calculate perplexity
    M = len(sentences[0].split(' ')) + len(sentences[1].split(' '))
    l = sum(probability_lst)
    l /= M
    perplexity = np.power(math.e, -l)

    return probability_lst, perplexity


def run():
    dict_ = parser()

    # Q2 continue the following sentence
    sentence_for_Q2 = "I have a house in"
    print("Q2, next word is: " + str(
        complete_the_sentence(sentence_for_Q2, dict_)))

    # Q3 compute the probability of the following two sentences
    first_sentence = "Brad Pitt was born in Oklahoma"
    second_sentence = "The actor was born in USA"
    sentences = [first_sentence, second_sentence]

    probability, perplexity = bigram_probability_perplexity(sentences, dict_)
    for i, sentence_prob in enumerate(probability):
        print("Q3.a: sentence " + str(i + 1) + " probability is: ")
        print(sentence_prob, end="\n")

    print("Q3.b: perplexity of the sentences is: ")
    print(perplexity, end="\n")

    # Q4 calculate linear interpolation smoothing:
    probability, perplexity = linear_interpolation_smoothing(sentences, dict_)
    for i, sentence_prob in enumerate(probability):
        print("Q4: linear interpolation smoothing of sentence " + str(i + 1)
              + " probability is: ")
        print(probability[i], end="\n")
    print("Q4: linear interpolation smoothing of sentence perplexity is: ")
    print(perplexity, end="\n")


if __name__ == '__main__':
    run()
