from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("autism_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from the webpage
    age = float(request.form["age"])
    gender = request.form["gender"]

    A1 = int(request.form["A1_Score"])
    A2 = int(request.form["A2_Score"])
    A3 = int(request.form["A3_Score"])
    A4 = int(request.form["A4_Score"])
    A5 = int(request.form["A5_Score"])
    A6 = int(request.form["A6_Score"])
    A7 = int(request.form["A7_Score"])
    A8 = int(request.form["A8_Score"])
    A9 = int(request.form["A9_Score"])
    A10 = int(request.form["A10_Score"])

    # Convert gender to number
    if gender == "m":
        gender_value = 1
    else:
        gender_value = 0

    # Create input data
    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender_value,
        "A1_Score": A1,
        "A2_Score": A2,
        "A3_Score": A3,
        "A4_Score": A4,
        "A5_Score": A5,
        "A6_Score": A6,
        "A7_Score": A7,
        "A8_Score": A8,
        "A9_Score": A9,
        "A10_Score": A10
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    if prediction == 1:
        result = "Likely ASD Screening Result"
    else:
        result = "Unlikely ASD Screening Result"

    return render_template(
        "result.html",
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)