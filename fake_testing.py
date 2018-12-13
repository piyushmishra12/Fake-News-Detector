import pandas as pd
import numpy as np
import sys

try:
    news_clean_encoded = np.loadtxt('test_news_encoding.gz')
except:
    print ("Please first encode the data!!")
    sys.exit(0)

try:
    model_file = open('clf_architecture.json', 'r')
except:
    print ("Save the trained model before testing!!")
    sys.exit(0)

from keras.models import model_from_json
loaded_model = model_file.read()
clf = model_from_json(loaded_model)
clf.load_weights('clf_weights.h5')

clf.compile(optimizer = 'adam', loss = 'binary_crossentropy')
x_test = news_clean_encoded

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
x_test = sc.fit_transform(x_test)

predictions = clf.predict(x_test)
