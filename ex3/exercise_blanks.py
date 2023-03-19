import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm
import matplotlib.pyplot as plt

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

# train nn parameters
N_EPOCHS = 20
BATCHES_SIZE = 64
LEARNING_RATE_PARAM = 0.01
W = 0.001

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"
NEGATED = "negated"
RARE = "rare"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """
    Load Word2Vec Vectors
    Return: wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    sum_array = np.zeros(embedding_dim)
    count = 0
    for word in sent.text:
        if word in word_to_vec.keys():
            count += 1
            sum_array += word_to_vec[word]
    return sum_array / (1 if not count else count)


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    one_hot_vec = np.zeros(size)
    one_hot_vec[ind] = 1
    return one_hot_vec


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return: average_array
    """
    size = len(word_to_ind.keys())
    sum_array = np.sum(
        get_one_hot(size, word_to_ind[word]) for word in sent.text)
    sum_ = sum(sum_array)
    return sum_array / (1 if not sum_ else sum_)


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    return {word: ind for ind, word in enumerate(words_list)}


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    sent_embedding = np.zeros((seq_len, embedding_dim))
    sent = sent.text
    len_sent = len(sent)
    if len_sent > seq_len:
        sent = sent[:seq_len]
    for i, word in enumerate(sent):
        if word in word_to_vec.keys():
            sent_embedding[i] = word_to_vec[word]
        else:
            sent_embedding[i] = np.zeros(embedding_dim)
    for i in range(len_sent, seq_len):
        sent_embedding[i] = np.zeros(embedding_dim)
    return sent_embedding


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True,
                 dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(
            dataset_path,
            split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[
                TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # add special subsets to dictionary
        self.sentences[NEGATED] = np.array(self.sentences[TEST])[
            data_loader.get_negated_polarity_examples(self.sentences[TEST])]
        self.sentences[RARE] = np.array(self.sentences[TEST])[
            data_loader.get_rare_words_examples(self.sentences[TEST],
                                                self.sentiment_dataset)]

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {
                "word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(
                                         words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(
                self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {
                "word_to_vec": create_or_load_slim_w2v(words_list),
                "embedding_dim": embedding_dim
            }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {
            k: OnlineDataset(sentences, self.sent_func,
                             self.sent_func_kwargs)
            for
            k, sentences in self.sentences.items()}
        self.torch_iterators = {
            k: DataLoader(dataset, batch_size=batch_size,
                          shuffle=k == TRAIN)
            for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array(
            [sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.lstm_layer = nn.LSTM(input_size=embedding_dim,
                                  hidden_size=hidden_dim, num_layers=n_layers,
                                  dropout=dropout, bidirectional=True,
                                  batch_first=True)
        self.linear_layer = nn.Linear(in_features=2 * hidden_dim,
                                      out_features=1)

    def forward(self, text):
        output, (hn, cn) = self.lstm_layer(text.to(torch.float32))
        concate_output = torch.cat((hn[0], hn[1]), dim=1)
        return self.linear_layer(concate_output)

    def predict(self, text):
        s = nn.Sigmoid()
        return s(self.forward(text))


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.first_layer = nn.Linear(in_features=embedding_dim, out_features=1)

    def forward(self, x):
        return self.first_layer(x.to(torch.float32))

    def predict(self, x):
        s = nn.Sigmoid()
        return s(self.forward(x))

    # ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    correct = torch.sum((preds >= 0.5) == y)  # , dtype=torch.float64
    return (correct / len(y)).item()


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    model.train()
    acc, loss = 0, None
    for sent, tag in data_iterator:
        optimizer.zero_grad()  # Zero the gradient
        prediction = model(sent)  # Forward pass
        loss = criterion(prediction[:, 0], tag)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights
        acc += binary_accuracy(prediction[:, 0], tag)  # Update accuracy
    return acc / len(data_iterator), loss.item()


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    model.eval()
    acc, loss = 0, None
    for sent, tag in data_iterator:
        prediction = model(sent)  # Forward pass
        loss = criterion(prediction[:, 0], tag)  # Compute the loss
        acc += binary_accuracy(prediction[:, 0], tag)  # Update accuracy
    return acc / len(data_iterator), loss.item()


def get_predictions_for_data(model, data_iter):
    """
    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return: prediction_array, accuracy, Loss
    """
    acc, loss = 0, 0
    criterion = nn.BCEWithLogitsLoss()
    prediction_arr = []
    for sent, tag in data_iter:
        predict = model.predict(sent)
        prediction_arr.append(predict)
        loss = criterion(predict[:, 0], tag)
        acc += binary_accuracy(predict[:, 0], tag)  # Update accuracy
    return prediction_arr, acc / len(data_iter), loss.item()


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    optimizer = optim.Adam(params=model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    train_acc_arr, train_loss_arr, val_acc_arr, val_loss_arr = [], [], [], []
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(n_epochs):
        train_acc, train_loss = train_epoch(model,
                                            data_manager.get_torch_iterator(
                                                TRAIN),
                                            optimizer, criterion)
        val_acc, val_loss = evaluate(model,
                                     data_manager.get_torch_iterator(VAL),
                                     criterion)
        train_acc_arr.append(train_acc)
        train_loss_arr.append(train_loss)
        val_acc_arr.append(val_acc)
        val_loss_arr.append(val_loss)

        # print validation accuracy todo is needed
        print(f'epoch {epoch} validation accuracy {val_acc}')
    return train_acc_arr, train_loss_arr, val_acc_arr, val_loss_arr


def train_log_linear_with_one_hot():
    """
    Train a log linear model
    """
    DM = DataManager(batch_size=BATCHES_SIZE)
    LL_model = LogLinear(DM.get_input_shape()[0])
    train = train_model(LL_model, DM, N_EPOCHS, LEARNING_RATE_PARAM, W)
    # for set_ in [RARE, NEGATED, TEST]:
    #     set_pred, set_acc, set_loss = get_predictions_for_data(LL_model,
    #                                                DM.get_torch_iterator(
    #                                                                set_))
    #     print(f"Log Linear model with one hot, {set_} accuracy is: " + str(
    #         set_acc))
    #     print(f"Log Linear model with one hot, {set_} Loss is: " + str(
    #         set_loss))
    return train


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    DM = DataManager(batch_size=BATCHES_SIZE, data_type=W2V_AVERAGE,
                     embedding_dim=300)
    LL_model = LogLinear(DM.get_input_shape()[0])

    train = train_model(LL_model, DM, N_EPOCHS, LEARNING_RATE_PARAM, W)
    # for set_ in [RARE, NEGATED, TEST]:
    #     set_pred, set_acc, set_loss = get_predictions_for_data(LL_model,
    #                                                  DM.get_torch_iterator(
    #                                                      set_))
    #     print(f"Log Linear model with w2v, {set_} accuracy is: " + str(
    #         set_acc))
    #     print(f"Log Linear model with w2v, {set_} Loss is: " + str(
    #         set_loss))
    return train


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    DM = DataManager(batch_size=BATCHES_SIZE, data_type=W2V_SEQUENCE,
                     embedding_dim=300)
    LSTM_model = LSTM(embedding_dim=300, hidden_dim=100, n_layers=1,
                      dropout=0.5)
    train = train_model(LSTM_model, DM, n_epochs=4, lr=0.001,
                        weight_decay=0.0001)
    # for set_ in [RARE, NEGATED, TEST]:
    #     set_pred, set_acc, set_loss = get_predictions_for_data(LSTM_model,
    #                                                  DM.get_torch_iterator(
    #                                                      set_))
    #     print(f"LSTM model with w2v, {set_} accuracy is: " + str(
    #         set_acc))
    #     print(f"LSTM model with w2v, {set_} Loss is: " + str(
    #         set_loss))
    return train


def plot(train_acc_arr, train_loss_arr, val_acc_arr, val_loss_arr, title,
         epoch):
    """
    plot accuracy and loss of train and validation sets
    :param train_acc_arr: accuracy of train set
    :param train_loss_arr: loss of train set
    :param val_acc_arr: accuracy of validation set
    :param val_loss_arr: loss of validation set
    :param title: plot's title
    """
    axis_range = np.arange(epoch)

    # create a accuracy plot
    plt.plot(axis_range, train_acc_arr, label='train accuracy', color="blue")
    plt.plot(axis_range, val_acc_arr, label='val accuracy', color="red")
    plt.xlabel("num of epoch")
    plt.ylabel("accuracy rate")
    plt.title(title + " - accuracy")
    plt.legend()
    plt.show()

    # create a loss plot
    plt.plot(axis_range, train_loss_arr, label='train loss', color="blue")
    plt.plot(axis_range, val_loss_arr, label='val loss', color="red")
    plt.xlabel("num of epoch")
    plt.ylabel("Loss rate")
    plt.title(title + " - Loss")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # train models
    hv_train_acc, hv_train_loss, hv_val_acc, \
    hv_val_loss = train_log_linear_with_one_hot()
    plot(hv_train_acc, hv_train_loss, hv_val_acc, hv_val_loss,
         "Log Linear - one hot vector", N_EPOCHS)


    w2v_train_acc, w2v_train_loss, \
    w2v_val_acc, w2v_val_loss = train_log_linear_with_w2v()
    plot(w2v_train_acc, w2v_train_loss, w2v_val_acc, w2v_val_loss,
         "Log Linear - word 2 vec", N_EPOCHS)

    lstm_train_acc, lstm_train_loss, \
    lstm_val_acc, lstm_val_loss = train_lstm_with_w2v()
    plot(lstm_train_acc, lstm_train_loss, lstm_val_acc, lstm_val_loss,
         "LSTM - word 2 vec", 4)
