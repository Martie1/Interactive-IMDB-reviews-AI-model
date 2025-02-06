import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#pre trained model with vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  
    text = text.lower()  
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def predict_sentiment(review):
    cleaned_review = preprocess_text(review) #clean review, no stop words.
    print("This is cleaned review: ",cleaned_review)
    transformed_review = vectorizer.transform([cleaned_review])  # conversion to tf-idf
    prediction = model.predict(transformed_review)[0]  # predict using our model
    
    if(prediction==1):
        return "Positive 😊"
    else:
        return "Negative 😠"

while True:
    user_review = input("\nEnter a movie review (or type 'exit' to quit): ")
    if user_review.lower() == 'exit':
        break
    sentiment = predict_sentiment(user_review)
    print(f"Predicted Sentiment: {sentiment}")