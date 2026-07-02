import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

import joblib

data = pd.read_csv("dataset.csv")

X = data["text"]

y = data["label"]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(
X,y,test_size=0.2,random_state=42
)

model = MultinomialNB()

model.fit(X_train,y_train)

prediction = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,prediction))

joblib.dump(model,"phishing_model.pkl")

joblib.dump(vectorizer,"vectorizer.pkl")

print("Model Saved Successfully")
