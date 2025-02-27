from flask import Flask, render_template, request
import pickle

# Load the trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        user_input = request.form["text"]
        data = vectorizer.transform([user_input]).toarray()
        prediction = model.predict(data)[0]
        return render_template("index.html", language=prediction)

if __name__ == "__main__":
    app.run(debug=True)
