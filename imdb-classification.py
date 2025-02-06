import pandas as pd

df = pd.read_csv("IMDB Dataset.csv")

print("-----HEAD")
print(df.head())
print("-----FIRST REVIEW")
print(df['review'][0])  # First review for insight

print("-----INFO")
print(df.info())

print("-----SENTIMENT VALUE counts")
print(df['sentiment'].value_counts()) 

import seaborn as sns
import matplotlib.pyplot as plt


#2 plot display
plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)  # (1 row, 2 columns, 1st plot)
sns.countplot(x=df['sentiment'], palette="viridis")
plt.title("Number of positive and negative reviews")

#mapping sentiment to binary
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  
    text = text.lower()  
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

#cleaning reviews from insignificant words like "the", "is", "of" etc.
#Focus on value bringing, meaningful words
df['clean_review'] = df['review'].apply(preprocess_text)

#TFidfVectorizer does the tokenizing process
vectorizer = TfidfVectorizer(max_features=5000)  #selection of only 5000 most common words in those reviews
X = vectorizer.fit_transform(df['clean_review']).toarray()  
y = df['sentiment'].values 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)
print("Accuracy :", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)


plt.subplot(1, 2, 2)  # (1 row, 2 columns, 2nd plot)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel("Prediction")
plt.ylabel("Real")
plt.title("Confusion Matrix")

plt.tight_layout()
plt.show()







