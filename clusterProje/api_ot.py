from fastapi import FastAPI
from pydantic import BaseModel
import re
import string
import numpy as np
import contractions
import joblib
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from sentence_transformers import SentenceTransformer
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


# NLP bileşenleri
stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

# FastAPI uygulaması
app = FastAPI(title="Clothing Review Clustering API - Auto Label")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Model yükleme
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
kmeans_model = joblib.load("kmeans_model.joblib")
auto_labels = joblib.load("auto_labels.joblib")  # Otomatik etiketler

# POS dönüşüm fonksiyonu
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Metin temizleme
def clean_text(text):
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = word_tokenize(text)
    words = [t for t in words if t not in stop_words and t not in punctuations and t.isalpha()]
    pos_tags = pos_tag(words)
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return " ".join(lemmatized)

# İstek modeli
class ReviewRequest(BaseModel):
    review_text: str

# Yanıt modeli
class PredictionResponse(BaseModel):
    predicted_cluster: int
    predicted_category: str
    cleaned_text: str

# Tahmin endpoint'i
@app.post("/predict", response_model=PredictionResponse)
def predict_category(req: ReviewRequest):
    cleaned = clean_text(req.review_text)
    emb = embedding_model.encode([cleaned], convert_to_numpy=True)
    cluster = int(kmeans_model.predict(emb)[0])
    category = auto_labels.get(cluster, "Unknown")

    return PredictionResponse(
        predicted_cluster=cluster,
        predicted_category=category,
        cleaned_text=cleaned
    )
#uvicorn api:app --reload
    

