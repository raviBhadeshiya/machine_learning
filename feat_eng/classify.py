from csv import DictReader, DictWriter
from collections import defaultdict
import random
import numpy as np
from numpy import array
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer


import pickle
import omdb,re

omdb.set_default('apikey','68e1c331') # Courtesy of peer for key sharing

train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
pages = [x['page'] for x in train]
unique_pages = set(pages)
genre_dict = defaultdict()

for page in unique_pages:
    page_title = re.sub(r'([A-Z][a-z]+)', r' \1', page).strip()
    try:
        movie = omdb.title(page_title)
        genre = movie['genre']
        genre_dict[page] = genre
    except:
        print("Exception:", page)
        genre_dict[page] = [""]
    print('##PageTitle:{} ##Genre:{}:'.format(page, genre_dict[page]))

# pickle.dump(genre_dict, open("genre_dict.p", "wb"))
## Load the Genre Dict
# genre_dict = pickle.load(open("genre_dict.p", "rb"))
## Return the Genre as per page
def findGenre(page):
    if page in genre_dict: return genre_dict[page]
    else: return [""]

class LemmaToken(object):
    def __init__(self):
        # self.wnl = WordNetLemmatizer()
        self.wnl = PorterStemmer()
        self.tokenizer = RegexpTokenizer(r'\b\w+\b')
        # self.tokenizer = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
    def __call__(self, doc):
        return [self.wnl.stem(t) for t in doc]
        # return [self.wnl.stem(t) for t in self.tokenizer.tokenize(doc)]
        # return [self.wnl.stem(t) for t in self.tokenizer.tokenize(doc)]
        # return [self.wnl.lemmatize(i, pos=j[0].lower()) if j[0].lower() in ['r', 'n', 'v']
        #         else self.wnl.lemmatize(i) for i, j in pos_tag(self.tokenizer.tokeni
        # ze(doc))]
        # return [self.wnl.lemmatize(i, pos=j[0].lower()) if j[0].lower() in ['a', 's', 'r', 'n', 'v']
        #         else self.wnl.lemmatize(i) for i, j in pos_tag(self.tokenizer.tokenize(doc))]

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'

class Featurizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range = (1, 4) ,
                                          # min_df = 2,
                                          # max_features = 50000,
                                          max_df=0.9,
                                          analyzer = 'word' ,
                                          # stop_words = 'english' ,
                                          # strip_accents = 'ascii' ,
                                          token_pattern=r'\b\w+\b',
                                          tokenizer = LemmaToken()
                                          )
        # self.vectorizer = FeatureUnion([
        #     ('main', Pipeline([
        #         ('selector', ItemSelector(kTEXT_FIELD)),
        #         ('vec', TfidfVectorizer(ngram_range=(1, 2),
        #                                 min_df=2,
        #                                 # max_features = 20000 ,
        #                                 analyzer='word',
        #                                 # stop_words='english',
        #                                 # strip_accents='ascii',
        #                                 token_pattern=r'\b\w+\b',
        #                                 tokenizer=LemmaToken())),
        #         ])
        #      ),
        #     ('genre', Pipeline([
        #         ('selector', ItemSelector('genre')),
        #         ('vec', CountVectorizer()),
        #         # ('to_dense', DenseTransformer()),
        #         # ('oneHot', OneHotEncoder())
        #     ])
        #     ),
        #     # ('page', Pipeline([
        #     #     ('selector', ItemSelector('page')),
        #     #     ('vec', CountVectorizer(analyzer='word',
        #     #                             min_df=2,
        #     #                             token_pattern=r'\b\w+\b'))
        #     #     ])
        #     # ),
        #     # ('verb', Pipeline([
        #     #     ('selector', ItemSelector('verb')),
        #     #     ('vec', CountVectorizer(analyzer='word',
        #     #                             token_pattern=r'\b\w+\b')),
        #     #     ])
        #     # ),
        #     # ('trope', Pipeline([
        #     #     ('selector', ItemSelector('trope')),
        #     #     ('vec', CountVectorizer(analyzer='word',
        #     #                             min_df=2,
        #     #                             token_pattern=r'\b\w+\b'))
        #     #     ])
        #     # )
        # ])

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-10:]
            bottom10 = np.argsort(classifier.coef_[0])[:10]
            print("Pos: %s" % " ".join(feature_names[top10]))
            print("Neg: %s" % " ".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))

if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))
    # random.shuffle(train)
    # train = train[:-int(len(train)*0.5)]
    # test = train[-int(len(train)*0.1):]
    # train = train[:-int(len(train)*0.1)]

    feat = Featurizer()

    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    print("Label set: %s" % str(labels))


    x_train = feat.train_feature(
        [" ".join([x[kTEXT_FIELD], "".join(findGenre(x['page'])), x['trope'], x['page']]) for x in train])
    x_test = feat.test_feature(
        [" ".join([x[kTEXT_FIELD], "".join(findGenre(x['page'])), x['trope'], x['page']]) for x in test])

    y_train = array(list(labels.index(x[kTARGET_FIELD])
                         for x in train))

    print(len(train), len(y_train))
    print(set(y_train))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)
    # sgd = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    # lr= Pipeline([('features',feat),('classifier',sgd)])
    
    print(accuracy_score(y_train,lr.predict(x_train)))
    # print(accuracy_score(y_test,lr.predict(x_test)))

    # feat.show_top10(lr, labels)

    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "cat"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'cat': labels[pp]}
        o.writerow(d)
