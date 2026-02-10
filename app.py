from flask import Flask, render_template, request
import numpy as np
import pickle

# Load model parameters
with open("linearModel.pkl", "rb") as f:
    model_data = pickle.load(f)

weights = model_data["weights"]
bias = model_data["bias"]

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [
        float(request.form['bedrooms']),
        float(request.form['bathrooms']),
        float(request.form['sqft_living']),
        float(request.form['sqft_lot']),
        float(request.form['floors']),
        float(request.form['waterfront']),
        float(request.form['view']),
        float(request.form['condition']),
        float(request.form['sqft_above']),
        float(request.form['sqft_basement']),
        float(request.form['yr_built']),
        float(request.form['yr_renovated']),
        float(request.form['year']),
        float(request.form['month']),
        float(request.form['day'])
    ]

    X = np.array(features)
    prediction = bias + np.dot(X, weights)

    return render_template(
        "index.html",
        prediction_text=f"₹ {prediction:,.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)
