from flask import Flask, render_template, flash, request, session,send_file
from flask import render_template, redirect, url_for, request

import sys

import pickle


import numpy as np

app = Flask(__name__)
app.config['DEBUG']
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

@app.route("/")
def homepage():
    return render_template('home.html')



@app.route("/result", methods=['GET', 'POST'])
def result():
    if request.method == 'POST':

        year = request.form['t1']
        month = request.form['t2']
        date = request.form['t3']







        filename = 'Model/prediction-rfc-model.pkl'
        classifier = pickle.load(open(filename, 'rb'))

        data = np.array([[year,month,date ]])

        my_prediction = classifier.predict(data)

        print(my_prediction)



        print(my_prediction[0])




        da = ("%.2f" % round(my_prediction[0], 2))

        return render_template('home.html',res=da)









if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)