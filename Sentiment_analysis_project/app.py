import nltk
nltk.download('stopwords')
from flask import Flask, request, jsonify
from flask_cors import CORS  
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import text
from keras.preprocessing.sequence import pad_sequences
import re
import os
word_tokenizer = text.Tokenizer()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 

current_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_directory, 'lstm_movie_review_model.keras')
model = load_model(model_path)

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
    print(sentence)
    return sentence

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data['review']
    processed_review = preprocess_text(review)
    tokenized_text = word_tokenizer.texts_to_sequences([processed_review])
    padded_text = pad_sequences(tokenized_text, padding='post', maxlen=100) 
    prediction = model.predict(padded_text)
    rating = prediction[0][0]
    
    return jsonify({'rating': str(rating)})

if __name__ == '__main__':
    app.run(debug=True)
