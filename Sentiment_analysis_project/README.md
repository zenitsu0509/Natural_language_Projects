<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
<!--     <title>Sentiment Analysis Web Application</title> -->
</head>
<body>


<h1>Sentiment Analysis Web Application</h1>

<p>This is a <strong>Sentiment Analysis Web Application</strong> built with <strong>Flask</strong>, <strong>TensorFlow/Keras</strong>, and <strong>NLP</strong> techniques. The application allows users to input text and predicts a sentiment rating using a pre-trained deep learning model.</p>

<h2>Features</h2>
<ul>
    <li>Sentiment prediction based on user input.</li>
    <li>Text preprocessing using NLTK and Keras tokenizer.</li>
    <li>Model trained using TensorFlow/Keras.</li>
    <li>RESTful API created with Flask.</li>
    <li>Cross-Origin Resource Sharing (CORS) enabled for secure API usage.</li>
</ul>

<h2>Technologies Used</h2>
<ul>
    <li><strong>Flask</strong>: Backend framework to serve the model and handle API requests.</li>
    <li><strong>TensorFlow/Keras</strong>: Used for training the model and performing predictions.</li>
    <li><strong>NLTK</strong>: Used for text preprocessing (stop word removal).</li>
    <li><strong>Pickle/JSON</strong>: Used for saving and loading tokenizers.</li>
</ul>

<h2>Project Structure</h2>
<pre><code>
├── app.py                     # Main Flask application
├── main.html                  # Frontend HTML file for the web interface
├── movie_review_project.ipynb  # Jupyter Notebook for movie review project
├── movie_review_project.py    # Python script for movie review project
├── my_model.keras             # Pre-trained Keras model
├── script.js                  # JavaScript file for frontend interactions
├── styles.css                 # CSS for frontend styling
├── word_tokenizer(1).json     # Saved tokenizer in JSON format
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
</code></pre>


<h2>Setup Instructions</h2>

<h3>1. Clone the Repository</h3>
<pre><code>
git clone https://github.com/your-repo/sentiment-analysis-app.git
cd sentiment-analysis-app
</code></pre>

<h3>2. Install Dependencies</h3>
<p>Ensure you have Python 3.7+ installed. You can install the required dependencies by running:</p>
<pre><code>
pip install -r requirements.txt
</code></pre>

<h3>3. Download NLTK Stopwords</h3>
<p>You need to download the NLTK stopwords for text preprocessing:</p>
<pre><code>
import nltk
nltk.download('stopwords')
</code></pre>

<h3>4. Run the Application</h3>
<p>Start the Flask app by running:</p>
<pre><code>
python app.py
</code></pre>
<p>The server will start at <a href="http://127.0.0.1:5000/" target="_blank">http://127.0.0.1:5000/</a>.</p>

<h2>API Usage</h2>

<h3>POST /predict</h3>
<p>You can use tools like Postman or cURL to send a POST request to the <code>/predict</code> endpoint with a JSON body containing the review text.</p>

<h4>Example Request:</h4>
<pre><code>
{
    "review": "This movie was absolutely fantastic!"
}
</code></pre>

<h4>Example Response:</h4>
<pre><code>
{
    "rating": "Positive"
}
</code></pre>

<h2>Model Training</h2>
<p>The sentiment analysis model was trained using a dataset of labeled reviews. The process involved:</p>
<ul>
    <li><strong>Text Preprocessing</strong>: Removal of stopwords, punctuation, etc.</li>
    <li><strong>Tokenization</strong>: Converting text to numerical format using the saved tokenizer.</li>
    <li><strong>Padding</strong>: Ensuring consistent input size for the model.</li>
    <li><strong>Neural Network Architecture</strong>: The model consists of embedding layers, LSTM layers, and dense layers for prediction.</li>
</ul>

<p>After training, both the model and the tokenizer were saved for future inference:</p>
<ul>
    <li><strong>Model</strong>: Saved as <code>my_model.keras</code>.</li>
    <li><strong>Tokenizer</strong>: Saved as <code>word_tokenizer.json</code> to avoid re-training when making predictions.</li>
</ul>

<h2>Known Issues</h2>
<ul>
    <li>Ensure that the tokenizer used for training is the same one used during prediction to avoid incorrect predictions.</li>
    <li>Differences in environments (e.g., TensorFlow versions) between the training setup (e.g., Colab) and the server might lead to varying predictions.</li>
</ul>

<h2>Future Improvements</h2>
<ul>
    <li>Implement more advanced NLP techniques such as BERT or Transformer models for better sentiment analysis.</li>
    <li>Improve the user interface by adding a front-end to interact with the web application.</li>
    <li>Allow users to upload documents for bulk sentiment analysis.</li>
</ul>

</body>
</html>
