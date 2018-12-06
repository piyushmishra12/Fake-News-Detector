import pandas as pd
import numpy as np
import string
import re
import os.path as op
from collections import Counter
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stopwords = stopwords.words('english')
df = pd.read_csv('fake_or_real_news.csv')

# Cleaning up the text
import spacy
nlp = spacy.load('en_core_web_sm')
punctuations = string.punctuation
def cleanup_text(docs, logging=False):
    texts = []
    counter = 1
    for doc in docs:
        if counter % 1000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)

# Encoding the clean text
news_clean_encoded = []
title_clean_encoded = []
vocab_size = 10000

from keras.preprocessing.text import hashing_trick
from keras.preprocessing.sequence import pad_sequences
def encode(text_list):
    encoded_list = []
    for i in range(6335):
        text = text_list[i]
        encoded = hashing_trick(text, vocab_size, hash_function = 'md5')
        encoded_list.append(encoded)
    return(encoded_list)

if op.isfile('news_encoding.gz'):
    news_clean_encoded = np.loadtxt('news_encoding.gz')
else:
    news = df.text
    news_clean = cleanup_text(news)
    news_clean_encoded = encode(news_clean)
    news_clean_encoded = pad_sequences(news_clean_encoded, maxlen = len(max(news_clean_encoded,
                                                                           key = len)))
    np.savetxt('news_encoding.gz',news_clean_encoded)

if op.isfile('title_encoding.gz'):
    title_clean_encoded = np.loadtxt('title_encoding.gz')
else:
    title = df.title
    title_clean = cleanup_text(title)
    title_clean_encoded = encode(title_clean)
    title_clean_encoded = pad_sequences(title_clean_encoded, maxlen = len(max(title_clean_encoded,
                                                                          key = len)))
    np.savetxt('title_encoding.gz',title_clean_encoded)

category = df.label
category = pd.get_dummies(category)
category = category.drop('REAL', 1)

category = category.values

x_train = news_clean_encoded

from keras.layers import Dense, Dropout
from keras.models import Sequential

clf = Sequential()

clf.add(Dense(units = 5312, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10623))

clf.add(Dense(units = 1000, kernel_initializer = 'uniform', activation = 'relu'))
#clf.add(Dropout(0.2))

clf.add(Dense(units = 500, kernel_initializer = 'uniform', activation = 'relu'))
clf.add(Dropout(0.25))

clf.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu'))
clf.add(Dropout(0.2))

clf.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

y_train = category

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
x_train = sc.fit_transform(x_train)

track = clf.fit(x_train, y_train, batch_size = 1000, epochs = 5)

import matplotlib.pyplot as plt
loss_fn = track.history['loss']
plt.plot(loss_fn)
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.show()

accuracy_fn = track.history['acc']
plt.plot(accuracy_fn)
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.show()
