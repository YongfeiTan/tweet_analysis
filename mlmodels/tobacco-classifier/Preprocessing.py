import nltk
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer

nltk.download('punkt')


# functions
def replace_pattern(text, replace_str, pattern):
    '''remove parts from the text according to the pattern

    Args:
        text: the text to be handled
        replace_str: the str used to replace the pattern
        pattern: a regular expression

    Returns:
        the tidy text after removing
    '''

    # re.findall() finds the pattern and puts it in a list for further task
    r = re.findall(pattern, str(text))

    # re.sub() replace patterns from the sentences in the dataset
    for i in r:
        text = text.replace(i, replace_str)

    return text


def lower_case(text):
    # make all words lower case
    text = text.lower()
    return text


def remove_stopwords(text):
    # remove natural language stop words in the text
    words = [w for w in text if w not in stopwords.words('english')]
    return words


def combine_text(list_of_word):
    return ' '.join(list_of_word)


def text_preprocessing(data):
    # apply all the NLP preprocessing
    print('preprocessing data...')
    data['text'] = data['text'].apply(lambda x: lower_case(x))
    print('finish lower case...')
    data['text'] = data['text'].apply(lambda x: replace_pattern(x, "", "@[\w]*"))
    print('remove tweets handles...')
    data['text'] = data['text'].apply(lambda x: replace_pattern(x, " ", "[^a-zA-Z#]"))
    print('remove punctuations and special characters...')
    data['text'] = data['text'].apply(lambda x: word_tokenize(x))
    print('finish tokenize...')
    data['text'] = data['text'].apply(lambda x: remove_stopwords(x))
    print('finish remove stop words...')
    ps = PorterStemmer()
    data['text'] = data['text'].apply(lambda x: [ps.stem(i) for i in x])
    print('stemming the tokens...')
    data['text'] = data['text'].apply(lambda x: combine_text(x))
    print('finish combine text...')

    # print(data['text'])

    return data

