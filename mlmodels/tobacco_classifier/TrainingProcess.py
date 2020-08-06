import numpy as np
import pandas as pd
import pickle

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import plot_confusion_matrix

from .Preprocessing import *

import matplotlib.pyplot as plt

class TobaccoTrainer(object):

    def __init__(self, dir_to_data):
        train_data = self.load_train_data(dir_to_data)
        self.count_vectorizer = self.vectorize_data(train_data)
        self.model = self.train(train_data)

        pickle.dump(self.count_vectorizer, open('tobacco_cv.sav', 'wb'))
        pickle.dump(self.model, open('tobacco_model.sav', 'wb'))

    @staticmethod
    def train_test_split(data, n=0.8):
        mask = np.random.rand(len(data)) < n
        train = data[mask]
        test = data[~mask]
        return train, test

    @staticmethod
    def vectorize_data(train):
        print('vectorizing data...')
        # change text to vector
        count_vectorizer = CountVectorizer(analyzer='word', binary=True)
        count_vectorizer.fit(train['text'])

        return count_vectorizer

    def load_train_data(self, dir_to_data):
        print('loading training data from disk...')
        data = pickle.load(open(dir_to_data, 'rb'))
        return data

    def train(self, train_data):
        model = SVC()
        data = text_preprocessing(train_data)

        train_data, test_data = TobaccoTrainer.train_test_split(data)

        train_vector = self.count_vectorizer.transform(train_data['text'])
        test_vector = self.count_vectorizer.transform(test_data['text'])

        y_train = train_data['target']
        y_test = test_data['target']
        print('start training...')
        scores = model_selection.cross_val_score(model, train_vector, y_train, cv=5, scoring="f1")

        print('finish. ')
        print('5 cross validation f1 scores: ', scores)
        model.fit(train_vector, y_train)

        titles_options = [("Confusion matrix, without normalization", None),
                          ("Normalized confusion matrix", 'true')]
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(model, test_vector, y_test,
                                         display_labels=['irrelevant', 'relevant'],
                                         cmap=plt.cm.Blues,
                                         normalize=normalize)
            disp.ax_.set_title(title)

            print(title)
            print(disp.confusion_matrix)

        plt.show()

        return model

if __name__ == '__main__':
    Trainer = TobaccoTrainer(r'D:\Study\Summer2020\mix_data_df.sav')