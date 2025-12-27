from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model & scaler
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = ""

    if request.method == "POST":
        data = [
            int(request.form["age"]),
            int(request.form["sex"]),
            int(request.form["cp"]),
            int(request.form["trestbps"]),
            int(request.form["chol"]),
            int(request.form["fbs"]),
            int(request.form["restecg"]),
            int(request.form["thalach"]),
            int(request.form["exang"]),
            float(request.form["oldpeak"]),
            int(request.form["slope"]),
            int(request.form["ca"]),
            int(request.form["thal"])
        ]

        columns = [
            'age','sex','cp','trestbps','chol','fbs',
            'restecg','thalach','exang','oldpeak',
            'slope','ca','thal'
        ]

        input_df = pd.DataFrame([data], columns=columns)
        input_scaled = scaler.transform(input_df)
        result = model.predict(input_scaled)

        if result[0] == 1:
            prediction_text = "⚠️ Heart Disease Detected"
        else:
            prediction_text = "✅ No Heart Disease Detected"

    return render_template("index.html", prediction=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
