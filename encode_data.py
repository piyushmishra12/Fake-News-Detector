import pandas as pd
import numpy as np
import string
import re
import preprocessor as pr
import contractions as con
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
        pr.set_options(pr.OPT.URL, pr.OPT.EMOJI, pr.OPT.HASHTAG, pr.OPT.MENTION, pr.OPT.RESERVED, pr.OPT.SMILEY, pr.OPT.NUMBER)
        doc = pr.clean(doc)
        doc = con.fix(doc)
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations and re.sub("[0-9]*",'',tok) != '']
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)
 
news = df.text
news_clean = cleanup_text(news)

title = df.title
title_clean = cleanup_text(title)

# Encoding the clean text
news_clean_encoded = []
title_clean_encoded = []
vocab_size = 10000

from keras.preprocessing.text import hashing_trick
def encode(text_list):
    encoded_list = []
    for i in range(len(text_list)):
        text = text_list[i]
        encoded = hashing_trick(text, vocab_size, hash_function = 'md5')
        encoded_list.append(encoded)
    return(encoded_list)

from keras.preprocessing.sequence import pad_sequences
news_clean_encoded = encode(news_clean)
news_clean_encoded = pad_sequences(news_clean_encoded, len(max(news_clean_encoded,
                                                                           key = len)))

title_clean_encoded = encode(title_clean)
title_clean_encoded = pad_sequences(title_clean_encoded, maxlen = len(max(title_clean_encoded,
                                                                          key = len)))

# Saving the encoded data
np.savetxt('news_encoding.gz',news_clean_encoded)
np.savetxt('title_encoding.gz',title_clean_encoded)
