###################################################
# Exercise 5 - Natural Language Processing 67658  #
###################################################

import numpy as np
import transformers
import sklearn
import matplotlib.pyplot as plt

# subset of categories that we will use
category_dict = {'comp.graphics': 'computer graphics',
                 'rec.sport.baseball': 'baseball',
                 'sci.electronics': 'science, electronics',
                 'talk.politics.guns': 'politics, guns'
                 }


def get_data(categories=None, portion=1.):
    """
    Get data for given categories and portion
    :param portion: portion of the data to use
    :return:
    """
    # get data
    from sklearn.datasets import fetch_20newsgroups
    data_train = fetch_20newsgroups(categories=categories, subset='train',
                                    remove=('headers', 'footers', 'quotes'),
                                    random_state=21)
    data_test = fetch_20newsgroups(categories=categories, subset='test',
                                   remove=('headers', 'footers', 'quotes'),
                                   random_state=21)

    # train
    train_len = int(portion * len(data_train.data))
    x_train = np.array(data_train.data[:train_len])
    y_train = data_train.target[:train_len]
    # remove empty entries
    non_empty = x_train != ""
    x_train, y_train = x_train[non_empty].tolist(), y_train[non_empty].tolist()

    # test
    x_test = np.array(data_test.data)
    y_test = data_test.target
    non_empty = np.array(x_test) != ""
    x_test, y_test = x_test[non_empty].tolist(), y_test[non_empty].tolist()
    return x_train, y_train, x_test, y_test


# Q1
def linear_classification(portion=1.):
    """
    Perform linear classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    tf = TfidfVectorizer(stop_words='english', max_features=1000)
    x_train, y_train, x_test, y_test = get_data(
        categories=category_dict.keys(), portion=portion)

    # TfidfVectorizer
    vectorizer = TfidfVectorizer()
    vec_x_train = vectorizer.fit_transform(x_train)
    vec_x_test = vectorizer.transform(x_test)

    # train the model on the training data
    clf = LogisticRegression(random_state=0)
    clf.fit(vec_x_train, y_train)

    # make predictions on the test data
    y_pred = clf.predict(vec_x_test)

    # evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


# Q2
def transformer_classification(portion=1.):
    """
    Transformer fine-tuning.
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    import torch

    class Dataset(torch.utils.data.Dataset):
        """
        Dataset object
        """

        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in
                    self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    from datasets import load_metric
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    from transformers import Trainer, TrainingArguments
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base',
                                              cache_dir=None)

    model = AutoModelForSequenceClassification.from_pretrained(
        'distilroberta-base',
        cache_dir=None,
        num_labels=len(category_dict),
        problem_type="single_label_classification")

    x_train, y_train, x_test, y_test = get_data(
        categories=category_dict.keys(), portion=portion)

    training_args = TrainingArguments(
        output_dir="/content/output",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5, )

    encodings_x_train = tokenizer(x_train, padding='longest', truncation=True)
    encodings_x_test = tokenizer(x_test, padding='longest', truncation=True)
    ds_x_train = Dataset(encodings_x_train, y_train)
    ds_x_test = Dataset(encodings_x_test, y_test)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_x_train,
        eval_dataset=ds_x_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics)

    trainer.train()
    pred_y = trainer.predict(ds_x_test)
    accuracy = trainer.evaluate()
    return accuracy["eval_accuracy"]


# Q3
def zeroshot_classification(portion=1.):
    """
    Perform zero-shot classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    from transformers import pipeline
    from sklearn.metrics import accuracy_score
    import torch
    x_train, y_train, x_test, y_test = get_data(
        categories=category_dict.keys(), portion=portion)
    clf = pipeline("zero-shot-classification",
                   model='cross-encoder/nli-MiniLM2-L6-H768',
                   device=torch.device(
                       'cuda:0' if torch.cuda.is_available() else 'cpu'))
    candidate_labels = list(category_dict.values())

    prediction_dict = {'computer graphics': 0, 'baseball': 1,
                       'science, electronics': 2, 'politics, guns': 3}

    pred = clf(x_test, candidate_labels=candidate_labels)

    y_pred = []
    for i in range(len(pred)):
        y_pred.append(prediction_dict[pred[i]['labels'][0]])

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def plot(portions, acc_arr, title):
    """
    Plot the model accuracy results as a function of the portion of the data.
    :param portions:
    :param acc_arr:
    :param title:
    """
    plt.plot(portions, acc_arr)
    plt.xlabel("portions")
    plt.ylabel("accuracy")
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    portions = [0.1, 0.5, 1.]

    # Q1
    acc_arr = []
    print("Logistic regression results:")
    for p in portions:
        print(f"Portion: {p}")
        acc = linear_classification(p)
        print(acc)
        acc_arr.append(acc)
    plot(portions, acc_arr,"Q1")

    # Q2
    acc_arr = []
    print("\nFinetuning results:")
    for p in portions:
        print(f"Portion: {p}")
        acc = transformer_classification(portion=p)
        print(acc)
        acc_arr.append(acc)
    plot(portions, acc_arr, "Q2")

    # # Q3
    print("\nZero-shot result:")
    print(zeroshot_classification())
