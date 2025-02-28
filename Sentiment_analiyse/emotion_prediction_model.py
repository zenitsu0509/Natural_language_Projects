import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
import pandas as pd
import string as str
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import string
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
import pandas as pd
import re
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

data = []
with open('/content/text.txt', 'r') as file:
    for line in file:
        array_part, text_part = line.split(']', 1)
        array = [float(x) for x in array_part.strip('[ ').split()]
        text = text_part.strip()
        data.append((array, text))
print("Array:", data[0])

df = pd.DataFrame(data, columns=['emotions', 'text'])

X = df['text']
y = np.array(df['emotions'].tolist())

def process_text(text):
    text = text.lower()
    text = ''.join([i for i in text if not i.isdigit()])
    text = ' '.join(text.split())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_words)
X_processed = [process_text(text) for text in X]

remove = ['.',',','(',')','\'','\"','\'','\""',]
for i in range(len(X_processed)):
  for j in remove:
    X_processed[i] = X_processed[i].replace(j,'')

for i in range(len(X_processed)):
  X_processed[i] = re.sub(' +', ' ',X_processed[i])

y = np.array(y)

target=['joy','fear','anger','sadness','disgust','shame','guilt']

labels = []
for v in y:
    labels.append(target[np.argmax(v)])
labels[:8]

labels = pd.DataFrame(labels)

labels.value_counts

labels.groupby(0).size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

labels.value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.ylabel('')
plt.title('Distribution of Emotions')

encoded_labels = label_encoder.fit_transform(labels)

from tensorflow.keras.utils import to_categorical
y_cat = to_categorical(encoded_labels)

y_cat

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(X_processed)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, encoded_labels, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(),
    'Multinomial Naive Bayes': MultinomialNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append({'Model': name, 'Accuracy': accuracy})

df_results = pd.DataFrame(results)
df_results

df_results.groupby('Model').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_tfidf, y_cat, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(100, activation='relu', input_dim=X_train_nn.shape[1]))
model.add(Dropout(0.6))
model.add(BatchNormalization())
model.add(Dense(7, activation='softmax'))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = model.fit(X_train_nn, y_train_nn, epochs=10, batch_size=32, validation_split=0.2)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

word = ['nice lets pparty']
word_processed = [process_text(text) for text in word]
word_tfidf = tfidf.transform(word_processed)

y_pred = model.predict(word_tfidf)

predicted_labels = np.argmax(y_pred, axis=1)
print([target[label] for label in predicted_labels])

model.summary()

