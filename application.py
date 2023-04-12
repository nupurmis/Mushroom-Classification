# -*- coding: utf-8 -*-
"""template_folder='template'
Created on Wed Feb 22 23:36:47 2023

@author: Lenovo
"""
from flask import Flask, request, app, render_template
import numpy as np
import pandas as pd
import os
import pickle

#from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__, template_folder='Templates', static_folder='static')
app = application

#Route for home page
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_data',methods=['GET','POST'])
def predict_data():
    print('Get started')
    if request.method=='GET':
        return render_template('home.html')
    else:
        print('Data input')
        data = [x for x in request.form.values()]
        input = np.array(data).reshape(1,-1)
        col=['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 
             'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 
             'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 
             'spore-print-color', 'population', 'habitat']

        pred_df=pd.DataFrame(data=input, columns=col)
        print('Data converted as data frame')
        print(pred_df)

        print("Before Prediction")
        results=predict(pred_df)
        print("after Prediction")
        print(results)

        result="Error"
        if results=='e':
            result="This mushroom is edible"
        else:
            result="This mushroom is poisonous"

        return render_template("home.html", prediction_text=result)

def predict(features):
    model_path=os.path.join("artifacts","model.pkl")
    preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

    print("Loading pickle files")
    model=pickle.load(open(model_path,'rb'))
    preprocessor=pickle.load(open(preprocessor_path,'rb'))

    print("Transforming data")
    data_processed=preprocessor.transform(features)

    print("Predicting output")
    preds=model.predict(data_processed)
    return preds

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=80,debug=True)
    