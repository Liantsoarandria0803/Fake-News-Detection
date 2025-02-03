from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from fastapi.middleware.cors import CORSMiddleware
nltk.download('punkt')
nltk.download('stopwords')


# Initialize app
app = FastAPI()

# Load the trained model and tokenizer
model = tf.keras.models.load_model('my_model.h5')  # Loading the Keras model
max_words = 10000  # Vocabulary size
max_length = 200  # Max sequence length
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")  # Initialize tokenizer

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Adjust this to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download NLTK data for tokenization and stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Text preprocessing function
def preprocess_text(text_data):
    preprocessed_text = []
    for sentence in text_data:
        # Remove non-alphanumeric characters
        sentence = re.sub(r'[^\w\s]', '', sentence)
        # Remove stopwords and convert to lowercase
        preprocessed_text.append(
            ' '.join(token.lower() for token in word_tokenize(sentence) if token not in stopwords.words('english'))
        )
    return preprocessed_text

# Request model
class NewsData(BaseModel):
    title: str
    text: str

@app.post("/predict/")
def predict_rain(data: NewsData):
    try:
        # Combine title and text into a single feature
        combined_text = data.title + " " + data.text
        # Preprocess the text
        preprocessed_text = preprocess_text([combined_text])
        
        # Tokenize and pad the preprocessed text
        tokenizer.fit_on_texts(preprocessed_text)  # Fit tokenizer on the preprocessed text (or load it if saved)
        sequences = tokenizer.texts_to_sequences(preprocessed_text)
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")
        
        # Predict using the model
        prediction = model.predict(padded_sequences)
        # Since this is a binary classification (fake news vs real news), 
        # we can use a threshold of 0.5 to classify the prediction.
        if prediction[0] >= 0.5:
            return {"prediction": "As my opinion, it's not fake news"}
        else:
            return {"prediction": "As my opinion, it's fake news"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
