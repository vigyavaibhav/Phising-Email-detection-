from flask import Flask,render_template,request

import joblib

model = joblib.load("phishing_model.pkl")

vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

@app.route("/",methods=["GET","POST"])

def home():

    result=""

    if request.method=="POST":

        email=request.form["email"]

        data=vectorizer.transform([email])

        prediction=model.predict(data)[0]

        result=prediction.upper()

    return render_template("index.html",result=result)

if __name__=="__main__":
    app.run(debug=True)
