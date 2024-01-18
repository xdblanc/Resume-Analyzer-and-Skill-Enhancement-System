# app.py

from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle
from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import math
import json

import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)
nlp = spacy.load('en_core_web_sm')
# Load the saved CNN model and tokenizer
model = load_model('cnn_model.h5')
with open('tokenizer.pkl', 'rb') as token_file:
    tokenizer = pickle.load(token_file)

import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras.utils import to_categorical
import pickle
from transformers import pipeline
# Load resume dataset
df = pd.read_csv('UpdatedResumeDataSet.csv')

# Handling missing values in 'Resume' column
df['Resume'] = df['Resume'].fillna('')  # Replace NaN values with an empty string

# Preprocess resume text
max_words = 5000
max_len = 200

tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
tokenizer.fit_on_texts(df['Resume'])
X = tokenizer.texts_to_sequences(df['Resume'])
X = pad_sequences(X, maxlen=max_len)

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Category'])
y = to_categorical(y)
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    all_text = []

    for page_num in range(num_pages):
        page = reader.pages[page_num]
        text = page.extract_text()
        all_text.append(text)

    return all_text

# Function to predict the category for a single resume text
def predict_category(resume_text):
    # Preprocess the input resume
    max_len = 200
    X_input = tokenizer.texts_to_sequences([resume_text])
    X_input = pad_sequences(X_input, maxlen=max_len)

 
    prediction = model.predict(X_input)[0]

    # Convert prediction to category label (modify as needed)
    predicted_label = label_encoder.inverse_transform([prediction.argmax()])[0]

    return predicted_label


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('punkt')

# Example texts
# job = "Machine learning is a subfield of artificial intelligence."
# x11 = "Artificial intelligence involves machines that can perform tasks that typically require human intelligence."

# Tokenize and remove stop words
def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return filtered_tokens



# Train Word2Vec model



# Function to vectorize a text using Word2Vec
def vectorize_text(text, model):
    vectorized_text = []
    for word in text:
        if word in model.wv:
            vectorized_text.append(model.wv[word])
    return np.mean(vectorized_text, axis=0)

# Vectorize texts using Word2Vec


# Calculate cosine similarity
def sim(x11,job):
    job_tokens = preprocess_text(job)
    x11_tokens = preprocess_text(x11)
    all_tokens = [job_tokens, x11_tokens]
    model = Word2Vec(all_tokens, vector_size=100, window=5, min_count=1, workers=4)
    vectorized_job = vectorize_text(job_tokens, model)
    vectorized_x11 = vectorize_text(x11_tokens, model)
    cosine_similarity_value = cosine_similarity([vectorized_job], [vectorized_x11])[0][0]
    match_percentage = cosine_similarity_value * 100
    matchpercentage=match_percentage
    return match_percentage





# Initialize the route for the main page
c=0
def score_resume(resume, phrases_and_scores):
    nlp = spacy.load("en_core_web_sm")
    global c
    # Process the resume text with spaCy
    doc = nlp(resume.lower())

    total_score = 0
    for phrase, score in phrases_and_scores.items():
        # Process the phrase with spaCy
        phrase_doc = nlp(phrase.lower())

        # Check for exact match
        if phrase_doc.text in doc.text:
            total_score += score
            c += 1
        else:
            # Check for similarity
            similarity = doc.similarity(phrase_doc)
            if similarity > 0.3:  # Adjust the threshold as needed
                total_score += score * similarity
                c = c + 1

    return math.ceil(total_score / c * 10)

def analyze_sentiment(text):
    sentiment_analyzer = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    result = sentiment_analyzer(text)
    return result[0]['label']




@app.route('/')
def index():
    return render_template('index.html')

# Initialize the route for analyzing the resume
@app.route('/analyze', methods=['POST'])
def analyze_resume():
    # Get the resume file or input from the form
    resume_file = request.files['resume']
    resume_text = request.form['resume_text']

    # Get the job description file or input from the form
    job_desc_file = request.files['job_desc']
    job_desc_text = request.form['job_desc_text']

    # Choose either file upload or manual input based on your preference
    if resume_file:
        x11 = extract_text_from_pdf(resume_file)
        x11 = ', '.join(x11)
    elif resume_text:
        x11 = resume_text
    else:
        return render_template('error.html', message='Please provide a resume.')

    # Choose either file upload or manual input based on your preference
    if job_desc_file:
        job_desc = extract_text_from_pdf(job_desc_file)
        job_desc = ', '.join(job_desc)
    elif job_desc_text:
        job_desc = job_desc_text
    else:
        return render_template('error.html', message='Please provide a job description.')

    # Predict the category
    predicted_category = predict_category(x11)

    # Calculate cosine similarity
    similarity = sim(x11, job_desc)


    skill_scores = {
    'python': 10,
    'machine learning': 8,
    'data analysis': 6,
    'communication skills': 8,
    'teamwork': 7,
    'java': 7,
    'cloud computing': 8,
    'problem solving': 9,
    'project management': 7,
    'sql': 7,
    'tensorflow': 9,
    'deep learning': 9,
    'leadership': 8,
    'agile': 8,
    'critical thinking': 9,
    'research': 7,
    'frontend development': 8,
    'backend development': 8,
    'web development': 8,
    'database design': 7,
    'networking': 7,
    'security': 8,
    'natural language processing': 9,
    'computer vision': 9,
    'data engineering': 8,
    'data visualization': 8,
    'statistics': 7,

    'technical writing': 7,
    'technical support': 6,
    'time management': 8,
    'multitasking': 7,
    
    'adaptability': 9,
    'strategic planning': 8,
    'collaboration': 8,
    'public speaking': 8,
    'mentoring': 8,

    
    'quality assurance': 7,
    'UX/UI design': 8,
    
    "bachelors degree": 5,
    "btech": 5,
    "mtech": 7,
    "masters": 7,
    "phd": 10,

    "Cisco Certified Network Associate (CCNA)": 7,
    "Cisco Certified Network Professional (CCNP)": 8,
    "Certified Information Systems Security Professional (CISSP)": 9,
    "Certified Ethical Hacker (CEH)": 8,
    "AWS Certified Solutions Architect": 8,
    "AWS Certified Developer": 7,
    "AWS Certified SysOps Administrator": 8,
    }

    result_score = score_resume(x11, skill_scores)
    print("aa")
    sentiment_result = analyze_sentiment(x11[:510])
    pipe = pipeline('sentiment-analysis')
    # print(x11[:400])
    senti=pipe(x11[:500])
    senti=senti[0]
    senti =  str(senti.get("label")) +"   "+ str(senti.get("score"))
    return render_template('result.html', predicted_category=predicted_category, similarity=similarity, result_score=result_score,sentiment_result=sentiment_result,senti=senti)


if __name__ == '__main__':
    app.run(debug=True)
