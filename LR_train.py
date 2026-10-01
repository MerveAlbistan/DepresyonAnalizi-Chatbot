import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

train_data = pd.read_csv("train_data.csv")

metin = train_data['text']
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(metin)

y_train = train_data['label']

model = LogisticRegression(class_weight='balanced')
model.fit(x_train, y_train)

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")