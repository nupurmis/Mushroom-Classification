# -*- coding: utf-8 -*-
"""template_folder='template'
Created on Wed Feb 22 23:36:47 2023

@author: Lenovo
"""
from flask import Flask, request, app, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__, template_folder='Templates', static_folder='static')
app = application

#Route for home page
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_data',methods=['POST'])
def predict_data():
    print('Get started')
    data=CustomData(
        cap_shape=request.form.get('cap-shape'),
        cap_surface=request.form.get('cap-surface'),
        cap_color=request.form.get('cap-color'),
        bruises=request.form.get('bruises'),
        odor=request.form.get('odor'),
        gill_attachment=request.form.get('gill-attachment'),
        gill_spacing=request.form.get('gill-spacing'),
        gill_size=request.form.get('gill-size'),
        gill_color=request.form.get('gill-color'),
        stalk_shape=request.form.get('stalk-shape'),
        stalk_root=request.form.get('stalk-root'),
        stalk_surface_above_ring=request.form.get('stalk-surface-above-ring'),
        stalk_surface_below_ring=request.form.get('stalk-surface-below-ring'),
        stalk_color_above_ring=request.form.get('stalk-color-above-ring'),
        stalk_color_below_ring=request.form.get('stalk-color-below-ring'),
        veil_type=request.form.get('veil-type'),
        veil_color=request.form.get('veil-color'),
        ring_number=request.form.get('ring-number'),
        ring_type=request.form.get('ring-type'),
        spore_print_color=request.form.get('spore-print-color'),
        population=request.form.get('population'),
        habitat=request.form.get('habitat')
    )
    print('Data created')
    pred_df=data.get_data_as_data_frame()
    print('Data converted as data frame')
    print(pred_df)
    print("Before Prediction")

    predict_pipeline=PredictPipeline()
    print("Mid Prediction")
    results=predict_pipeline.predict(pred_df)
    print("after Prediction")
    print(results)

    result="Error"
    if results=='e':
        result="This mushroom is edible"
    else:
        result="This mushroom is poisonous"

    return render_template("home.html", prediction_text=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port = 80)
    