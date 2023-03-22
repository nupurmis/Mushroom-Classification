# -*- coding: utf-8 -*-
"""template_folder='template'
Created on Wed Feb 22 23:36:47 2023

@author: Lenovo
"""
import json
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__,template_folder='Templates')
#Load the model
model = pickle.load(open('model.pkl','rb'))
enc = pickle.load(open('encoding.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])

def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
#    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [x for x in request.form.values()]
    input = np.array(data).reshape(1,-1)
    print(input)

    input = np.delete(input,15)
    print(input)  

    
    if input[16] =='n':
        input[16] = 0
    elif input[16] == 'o':
        input[16] = 1
    else:
        input[16] = 2

    print(input)  

    if ((input[4] =='a')|(input[4] =='l')):
        input[4] = 1
    elif input[4] == 'n':
        input[4] = 2
    else:
        input[4] = 0

    print(input)  

    odor = input[4]
    ring = input[16]
    input = np.delete(input,4)    
    input = np.delete(input,15)

    print(input)    

    input=np.array(input).reshape(1,-1)
    final_input=enc.transform(np.array(input).reshape(1,-1)).toarray()

    print(final_input)

    final_input=np.append(final_input,[odor])
    final_input=np.append(final_input,[ring])
    final_input=final_input.reshape(1,-1)
    print(final_input)
     
    final_input = [float(x) for x in final_input[0]]
    output = model.predict([final_input])
    print(output)
    
    result="Error"
    if output=='e':
        result="This mushroom is edible"
    else:
        result="This mushroom is poisonous"

    return render_template("home.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
    
