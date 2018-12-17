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

#Split Data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(news_clean_encoded, category, stratify=category, test_size = 0.20, random_state = 15)
input_dimen = x_train.shape[1]

# Model Definition
from keras.layers import Dense, Dropout
from keras.models import Sequential

clf = Sequential()

clf.add(Dense(units = input_dimen // 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = input_dimen))

clf.add(Dense(units = 1000, kernel_initializer = 'uniform', activation = 'relu'))
#clf.add(Dropout(0.2))

clf.add(Dense(units = 500, kernel_initializer = 'uniform', activation = 'relu'))
clf.add(Dropout(0.25))

clf.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu'))
clf.add(Dropout(0.2))

clf.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

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

#Evaluate Model
x_test = sc.fit_transform(x_test)
score = clf.evaluate(x = x_test, y = y_test, batch_size = 100, verbose = 1)

print ("Evaluation loss: " + str(score[0]))
print ("Evaluation acc: " + str(score[1]))

from sklearn.metrics import confusion_matrix
prediction = clf.predict(x_test)
labels = [0, 1]
con_mat = confusion_matrix(y_true = y_test, y_pred = np.round(prediction), labels = labels)

print ("Confusion Matrix")
print (con_mat)

from sklearn.metrics import classification_report
labels = ['Real', 'Fake']
print (classification_report(y_true = y_test, y_pred = np.round(prediction), target_names = labels))

from sklearn.metrics import roc_curve, auc
fal_pos_rate, tru_pos_rate, threshold = roc_curve(y_test, prediction)

area_under_curve = auc(fal_pos_rate, tru_pos_rate)
plt.figure()
plt.plot(fal_pos_rate, tru_pos_rate, label='AUC={:.3f}'.format(area_under_curve))
plt.legend(loc='best')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC curve')
plt.show()

#Save Model
model_file = open('clf_architecture.json','w')
clf.save_weights('clf_weights.h5')
model_file.write(clf.to_json())
