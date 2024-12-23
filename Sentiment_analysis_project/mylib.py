import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import re
import os
from tensorflow.keras.preprocessing.text import tokenizer_from_json

nltk.download('stopwords')

current_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_directory, 'my_model.keras')
tokenizer_path = os.path.join(current_directory, 'word_tokenizer(1).json')

model = load_model(model_path)
with open(tokenizer_path, 'r') as f:
    tokenizer_json = f.read()
    word_tokenizer = tokenizer_from_json(tokenizer_json)

target_tag = re.compile(r'<[^>]+>')
def remove_tags(text):
    return target_tag.sub('', text)

def preprocess_text(sen):
    sentence = sen.lower()
    sentence = remove_tags(sentence)
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    sentence = pattern.sub('', sentence)
    return sentence


def predict(request):
    data = request
    review = data['review']
    processed_review = preprocess_text(review)
    tokenized_text = word_tokenizer.texts_to_sequences([processed_review])
    padded_text = pad_sequences(tokenized_text, padding='post', maxlen=100) 
    prediction = model.predict(padded_text)
    rating = np.round(np.mean(prediction)*10,1)
    return {'rating': str(rating)}

