
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    inputs = [float(x) for x in request.form.values()]
    prediction = model.predict([inputs])
    result = "High Risk of Death" if prediction[0] == 1 else "Low Risk of Death"
    return render_template('index.html', prediction_text=f"Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True)
