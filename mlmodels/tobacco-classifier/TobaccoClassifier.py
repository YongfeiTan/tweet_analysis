import pickle
import pandas as pd

from .Preprocessing import *


class TobaccoClassifier(object):

    def __init__(self, cv_dir='tobacco_cv.sav', model_dir='tobacco_model.sav'):
        # model used to do classfication
        try:
            print('loading model from disk...')
            self.count_vectorizer = pickle.load(open(cv_dir, 'rb'))
            self.model = pickle.load(open(model_dir, 'rb'))

        except:
            print('error, no local model file found')

    def predict(self, test_data):
        test_data = text_preprocessing(test_data)
        test_vector = self.count_vectorizer.transform(test_data['text'])
        return self.model.predict(test_vector)


if __name__ == '__main__':
    TC = TobaccoClassifier()
    test_dict = {'Hello': None, 'Hemp':None}
    test_data = pd.DataFrame(test_dict.items(), columns = ['text', 'label'])
    result = TC.predict(test_data)
    print(result)
