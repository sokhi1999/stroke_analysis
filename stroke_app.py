from flask.helpers import url_for
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('model.pickle', 'rb'))
scaler = StandardScaler()


@app.route("/")
@app.route("/home")
def home_page():
    return render_template('homepage.html')

@app.route("/app")
def app_page():
    return render_template('app.html')

@app.route("/about")
def about_page():
    return render_template('about.html')

@app.route('/result', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        gender = request.form['gender']
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        disease = int(request.form['disease'])
        married = request.form['married']
        work = request.form['work']
        residence = request.form['residence']
        glucose = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking = request.form['smoking']

        input_features = [gender, age, hypertension, disease,
                          married, work, residence, glucose, bmi, smoking]

        features_value = [np.array(input_features)]
        features_name = ['gender', 'age', 'hypertension', 'disease',
                         'married', 'work', 'residence', 'glucose', 'bmi', 'smoking']

        df = pd.DataFrame(features_value, columns=features_name)
        prediction = model.predict(df)[0]
        if prediction == 1:
            return render_template('app.html', prediction_text='The patient has a likelihood of getting stroke.')
        else:
            return render_template('app.html', prediction_text='The patient does not have a likelihood of getting stroke.')




if __name__ == "__main__":
    app.run()