from csv import DictReader, DictWriter
import pickle
import random
import numpy as np
from numpy import array
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline,FeatureUnion

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer


class ItemSelector(TransformerMixin):
    def __init__(self,key):
        self.key = key
    def fit(self, X, y=None):
        return self
    def transform(self,data):
        return [x[self.key] for x in data]
import pickle
genre_dict = pickle.load(open("genre_dict.p", "rb"))

class PageGenre(TransformerMixin):
    def __init__(self, genre_dict):
        self.genre_dict = genre_dict

    def transform(self, X, **transform_params):
        # Given page, lookup genre
        genre_lists = [self.genre_dict[page] for page in X]
        genre_strings = [" ".join(genre_list) for genre_list in genre_lists]
        return genre_strings

    def fit(self, X, y=None, **fit_params):
        return self  # does nothing

class LemmaToken(object):
    def __init__(self):
        # self.wnl = WordNetLemmatizer()
        self.wnl = PorterStemmer()
        self.tokenizer = RegexpTokenizer(r'\w+')
        # self.tokenizer = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
    def __call__(self, doc):
        # return [self.wnl.stem(t) for t in doc]
        # return [self.wnl.lemmatize(t) for t in self.tokenizer.tokenize(doc)]
        return [self.wnl.stem(t) for t in self.tokenizer.tokenize(doc)]
        # return [self.wnl.lemmatize(i, pos=j[0].lower()) if j[0].lower() in ['r', 'n', 'v']
        #         else self.wnl.lemmatize(i) for i, j in pos_tag(self.tokenizer.tokeni
        # ze(doc))]
        # return [self.wnl.lemmatize(i, pos=j[0].lower()) if j[0].lower() in ['a', 's', 'r', 'n', 'v']
        #         else self.wnl.lemmatize(i) for i, j in pos_tag(self.tokenizer.tokenize(doc))]

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'

# class Featurizer:
#     def __init__(self):
#         self.vectorizer = TfidfVectorizer(ngram_range = (1, 2) ,
#                                           min_df = 2 ,
#                                           # max_features = 20000 ,
#                                           analyzer = 'word' ,
#                                           stop_words = 'english' ,
#                                           strip_accents = 'ascii' ,
#                                           # token_pattern=r'\b\w+\b',
#                                           tokenizer = LemmaToken())
#
#     def train_feature(self, examples):
#         return self.vectorizer.fit_transform(examples)
#
#     def test_feature(self, examples):
#         return self.vectorizer.transform(examples)
#
#     def show_top10(self, classifier, categories):
#         feature_names = np.asarray(self.vectorizer.get_feature_names())
#         print(len(feature_names))
#         if len(categories) == 2:
#             top10 = np.argsort(classifier.coef_[0])[-10:]
#             bottom10 = np.argsort(classifier.coef_[0])[:10]
#             print("Pos: %s" % " ".join(feature_names[top10]))
#             print("Neg: %s" % " ".join(feature_names[bottom10]))
#         else:
#             for i, category in enumerate(categories):
#                 top10 = np.argsort(classifier.coef_[i])[-10:]
#                 print("%s: %s" % (category, " ".join(feature_names[top10])))

if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    # test = list(DictReader(open("../data/spoilers/test.csv", 'r')))
    # random.shuffle(train)
    # p=0.10
    test = train[-int(len(train)*0.10):]
    train = train[:-int(len(train)*0.10)]

    # feat = Featurizer()
    feat = FeatureUnion([
        ('main',Pipeline([
            ('selector', ItemSelector(kTEXT_FIELD)),
            ('vec',TfidfVectorizer(ngram_range = (1, 2) ,
                                          min_df = 2 ,
                                          # max_features = 20000 ,
                                          analyzer = 'word' ,
                                          stop_words = 'english' ,
                                          strip_accents = 'ascii' ,
                                          # token_pattern=r'\b\w+\b',
                                          tokenizer = LemmaToken()))
            ])
        ),
        ('page', Pipeline([
            ('selector', ItemSelector('page')),
            ('page_lookup', PageGenre(genre_dict)),
            ('vec',CountVectorizer(analyzer = 'word'))
            ])
        ),
        ('trope', Pipeline([
            ('selector', ItemSelector('trope')),
            ('vec', CountVectorizer(analyzer='word'))
            ])
        ),
        # ('verb', Pipeline([
        #     ('selector', ItemSelector('verb')),
        #     ('vec', CountVectorizer(analyzer='word'))
        #     ])
        # )
    ])

    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    print("Label set: %s" % str(labels))
    # x_train = feat.train_feature(x[kTEXT_FIELD] for x in train)
    # x_test = feat.test_feature(x[kTEXT_FIELD] for x in test)
    # x_train = feat.train_feature([" ".join([x[kTEXT_FIELD],x['page'],x['trope']]) for x in train])
    # x_test = feat.test_feature([" ".join([x[kTEXT_FIELD],x['page'],x['trope']]) for x in test])
    x_train = train
    x_test = test
    y_train = array(list(labels.index(x[kTARGET_FIELD])
                         for x in train))
    y_test = array(list(labels.index(x[kTARGET_FIELD])
                         for x in test))
    # Train classifier
    # lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    sgd = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr= Pipeline([('features',feat),('classifier',sgd)])

    lr.fit(x_train, y_train)
    print(accuracy_score(y_train,lr.predict(x_train)))
    print(accuracy_score(y_test,lr.predict(x_test)))
    # feat.show_top10(lr, labels)