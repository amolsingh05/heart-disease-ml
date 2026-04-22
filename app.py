from flask import Flask, render_template, request
import pickle
import numpy as np

# create flask app
app = Flask(__name__)

# load trained model
model = pickle.load(open("heart_model.pkl", "rb"))

# home page
@app.route("/")
def home():
    return render_template("index.html")

# prediction
@app.route("/predict", methods=["POST"])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)

    prediction = model.predict(final_features)

    if prediction[0] == 1:
        result = "Heart Disease Detected"
    else:
        result = "No Heart Disease"

    return render_template("index.html", prediction_text=result)

# run server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
