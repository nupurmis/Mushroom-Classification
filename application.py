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

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        model_path=os.path.join("artifacts","model.pkl")
        preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
        print("Predict pipeline: Before Loading")
        model=pickle.load(open(model_path,'rb'))
        preprocessor=pickle.load(open(preprocessor_path,'rb'))
        print("After Loading")
        data_processed=preprocessor.transform(features)
        preds=model.predict(data_processed)
        return preds



class CustomData:
    def __init__( self,
        cap_shape: str,
        cap_surface: str,
        cap_color: str,
        bruises: str,
        odor: str,
        gill_attachment: str,
        gill_spacing: str,
        gill_size: str,
        gill_color: str,
        stalk_shape: str,
        stalk_root: str,
        stalk_surface_above_ring: str,
        stalk_surface_below_ring: str,
        stalk_color_above_ring: str,
        stalk_color_below_ring: str,
        veil_type: str,
        veil_color: str,
        ring_number: str,
        ring_type: str,
        spore_print_color: str,
        population: str,
        habitat: str
        ):

        self.cap_shape = cap_shape
        self.cap_surface = cap_surface
        self.cap_color = cap_color
        self.bruises = bruises
        self.odor = odor
        self.gill_attachment = gill_attachment
        self.gill_spacing = gill_spacing
        self.gill_size = gill_size
        self.gill_color = gill_color
        self.stalk_shape = stalk_shape
        self.stalk_root = stalk_root
        self.stalk_surface_above_ring = stalk_surface_above_ring
        self.stalk_surface_below_ring = stalk_surface_below_ring
        self.stalk_color_above_ring = stalk_color_above_ring
        self.stalk_color_below_ring = stalk_color_below_ring
        self.veil_type = veil_type
        self.veil_color = veil_color
        self.ring_number = ring_number
        self.ring_type = ring_type
        self.spore_print_color = spore_print_color
        self.population = population
        self.habitat = habitat
    
    def get_data_as_data_frame(self):
        custom_data_input_dict = {
            "cap-shape": [self.cap_shape],
            "cap-surface": [self.cap_surface],
            "cap-color": [self.cap_color],
            "bruises": [self.bruises],
            "odor": [self.odor],
            "gill-attachment": [self.gill_attachment],
            "gill-spacing": [self.gill_spacing],
            "gill-size": [self.gill_size],
            "gill-color": [self.gill_color],
            "stalk-shape": [self.stalk_shape],
            "stalk-root": [self.stalk_root],
            "stalk-surface-above-ring": [self.stalk_surface_above_ring],
            "stalk-surface-below-ring": [self.stalk_surface_below_ring],
            "stalk-color-above-ring": [self.stalk_color_above_ring],
            "stalk-color-below-ring": [self.stalk_color_below_ring],
            "veil-type": [self.veil_type],
            "veil-color": [self.veil_color],
            "ring-number": [self.ring_number],
            "ring-type": [self.ring_type],
            "spore-print-color": [self.spore_print_color],
            "population": [self.population],
            "habitat": [self.habitat]
        }

        return pd.DataFrame(custom_data_input_dict)


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=80,debug=True)
    