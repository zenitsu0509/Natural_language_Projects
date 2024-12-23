import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from numpy import array
from tensorflow.keras.preprocessing import text
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense
from keras.layers import Flatten, GlobalMaxPooling1D, Embedding, Conv1D, LSTM
from sklearn.model_selection import train_test_split
from keras.layers import Conv1D

import pandas as pd

# Attempt to read the file, handling potential errors
try:
    movie_reviews = pd.read_csv('/content/IMDB Dataset.csv')
except pd.errors.ParserError as e:
    print(f"Error reading CSV: {e}")
    # Inspect the problematic row and surrounding rows
    problematic_row_index = 35632
    with open('/content/IMDB Dataset.csv', 'r') as file:
        lines = file.readlines()
        for i in range(problematic_row_index - 2, problematic_row_index + 3):
            if 0 <= i < len(lines):
                print(f"Line {i+1}: {lines[i]}")

movie_reviews.head()

movie_reviews.isnull().any()

movie_reviews.shape

import seaborn as sns
sns.countplot(x='sentiment', data=movie_reviews)

movie_reviews["review"][5]

target_tag = re.compile(r'<[^>]+>')

def remove_tags(text):
    return target_tag.sub('', text)

import nltk
nltk.download('stopwords')

def preprocess_text(sen):
    sentence = sen.lower()
    sentence = remove_tags(sentence)
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    sentence = pattern.sub('', sentence)
    return sentence

X = []
sentences = list(movie_reviews['review'])
for sen in sentences:
    X.append(preprocess_text(sen))

X[5]

y = movie_reviews['sentiment']
y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

word_tokenizer = text.Tokenizer()
word_tokenizer.fit_on_texts(X_train)

X_train = word_tokenizer.texts_to_sequences(X_train)
X_test = word_tokenizer.texts_to_sequences(X_test)

vocab_length = len(word_tokenizer.word_index) + 1
vocab_length

maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

!kaggle datasets download -d sawarn69/glove6b100dtxt

!unzip glove6b100dtxt.zip

from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()
glove_file = open('/content/glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_length, 100))
for word, index in word_tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

embedding_matrix.shape

snn_model = Sequential()
embedding_layer = Embedding(vocab_length, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)

snn_model.add(embedding_layer)

snn_model.add(Flatten())
snn_model.add(Dense(1, activation='sigmoid'))

snn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

snn_model_history = snn_model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.2)

score = snn_model.evaluate(X_test, y_test, verbose=1)

import matplotlib.pyplot as plt

plt.plot(snn_model_history.history['acc'])
plt.plot(snn_model_history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(snn_model_history.history['loss'])
plt.plot(snn_model_history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

snn_model.summary()

cnn_model = Sequential()

embedding_layer = Embedding(vocab_length, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
cnn_model.add(embedding_layer)

cnn_model.add(Conv1D(128, 5, activation='relu'))
cnn_model.add(GlobalMaxPooling1D())
cnn_model.add(Dense(1, activation='sigmoid'))

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

cnn_model_history = cnn_model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1, validation_split=0.2)

score = cnn_model.evaluate(X_test, y_test, verbose=1)

plt.plot(cnn_model_history.history['acc'])
plt.plot(cnn_model_history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()

plt.plot(cnn_model_history.history['loss'])
plt.plot(cnn_model_history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()

cnn_model.summary()

lstm_model = Sequential()
embedding_layer = Embedding(vocab_length, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)

lstm_model.add(embedding_layer)
lstm_model.add(LSTM(128))

lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

lstm_model_history = lstm_model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

score = lstm_model.evaluate(X_test, y_test, verbose=1)

plt.plot(lstm_model_history.history['acc'])
plt.plot(lstm_model_history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(lstm_model_history.history['loss'])
plt.plot(lstm_model_history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

lstm_model.summary()

review = 'Best film ever This is the first sequel where I am not able to decide between part 1 and part 2 but still i would say Stree Part 1 is slightly better than this in terms of  comedyRajkumar again what a superb actor he is but I have no idea why he did plastic surgery on face looking bit weird but acting is as usual superb Rajkumar Rao is once again in top form, delivering his signature performance with strong support from Aparshakti Khurana and Abhishek Banerjee The first movie was a massive success and the sequel created an enormous buzz before its release special mention to Sachin Jigar music and Tammanah Bhatia dance too good '

review = preprocess_text(review)

unseen_tokenized = word_tokenizer.texts_to_sequences(review)
unseen_padded = pad_sequences(unseen_tokenized, padding='post', maxlen=maxlen)

y_pred = lstm_model.predict(unseen_padded)

rating =  np.round(y_pred*10,1)

print(np.mean(rating))

lstm_model.save('lstm_movie_review.keras')

