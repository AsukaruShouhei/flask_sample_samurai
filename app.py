#!/usr/bin/env python
# coding: utf-8
from flask import Flask, render_template
import pandas as pd
import iris_learn

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ayame')
def ayame_readcsv():
    df = pd.read_csv('csv/iris.csv')
    q = df.values.tolist()
    return render_template('ayamedata.html', data=q)


@app.route('/accuracy')
def accuracy():
    ress = iris_learn.getIris()
    return render_template('accuracy.html', ress=ress)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True, use_debugger=True)
