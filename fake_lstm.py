import pandas as pd
import numpy as np
import sys

# Importing the data
df = pd.read_csv('fake_or_real_news.csv')
try:
    news_clean_encoded = np.loadtxt('news_encoding.gz')
    title_clean_encoded = np.loadtxt('title_encoding.gz')
except:
    print ("Please first encode the data!!")
    sys.exit(0)

category = df.label
category = pd.get_dummies(category)
category = category.drop('REAL', 1)

category = category.values

x_train = news_clean_encoded
input_dimen = x_train.shape[1]

# Model Definition
from keras.layers import Dense, LSTM, Bidirectional
from keras.models import Sequential

lstm_seq_len = 40
clf = Sequential()

clf.add(Bidirectional(LSTM(units=64, kernel_initializer = 'uniform', return_sequences = True, input_shape = ((input_dimen-lstm_seq_len) // lstm_seq_len,lstm_seq_len))))

clf.add(Bidirectional(LSTM(units=32, kernel_initializer = 'uniform')))

clf.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

y_train = category

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
x_train = sc.fit_transform(x_train)
mod_x_train = []
for i in range(len(x_train)):
    mod_x_train.append([x_train[i][j:j+lstm_seq_len] for j in range(0,(input_dimen-lstm_seq_len),lstm_seq_len)])

x_train = np.array(mod_x_train)    
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
