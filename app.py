import pickle
import pandas as pd
import pgmpy
from sklearn.ensemble import RandomForestClassifier
from pgmpy.estimators import MaximumLikelihoodEstimator
import time
import sys
import re
from flask import Flask, jsonify, request, session
import psycopg2
from psycopg2 import sql
from validate_email_address import validate_email
import joblib
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
import numpy as np
from joblib import load
import torch
from PIL import Image
import matplotlib.pyplot as plt
import json
import requests

app = Flask(__name__)
app.secret_key = "uqdgyqUYGuqhaj23ndjqb"

db_config = {
    "host": "localhost",
    "user": "postgres",
    "password": "",
    "port": 5432,
    "dbname": "hasil_test",
}

def get_db_connection():
    return psycopg2.connect(**db_config)

def check_diabetes(evidence, model):
    columns_ordered = ['highbp', 'highchol', 'bmi', 'smoker', 'stroke', 'heartdisease', 'physactivity', 'diffwalk', 'sex', 'age']
    evidence_df = pd.DataFrame(evidence, columns=columns_ordered)
    result = model.predict(evidence_df)
    if result[0] == 1:
        return "Diabetes"
    else:
        return "Non-Diabetes"

loaded_model = load('modeldiabetes.joblib')

@app.route("/predict", methods=["POST"])
def predict_diabetes():
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    bmi = int(request.form['bmi'])
    highbp = int(request.form['highbp'])
    highchol = int(request.form['highchol'])
    smoker = int(request.form['smoker'])
    stroke = int(request.form['stroke'])
    heartdisease = int(request.form['heartdisease'])
    diffwalk = int(request.form['diffwalk'])
    physactivity = int(request.form['physactivity'])
    
    evidence = [[age, sex, bmi, highbp, highchol, smoker, stroke, heartdisease, diffwalk, physactivity]]
    result = check_diabetes(evidence, loaded_model)
    
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql_query = """
                INSERT INTO hasil_test (age, sex, bmi, highbp, highchol, smoker, stroke, heartdisease, diffwalk, physactivity, result)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql_query, (age, sex, bmi, highbp, highchol, smoker, stroke, heartdisease, diffwalk, physactivity, result))
            connection.commit()
    finally:
        connection.close()
    
    return jsonify({"Hasil Prediksi Risiko Diabetes": result})

@app.route("/hasil", methods=["GET"])
def get_hasil_test():
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql_query = "SELECT * FROM hasil_test WHERE username IS NULL" 
            cursor.execute(sql_query)
            hasil_test_data = cursor.fetchall()
            if hasil_test_data:
                return jsonify(hasil_test_data)
            else:
                return jsonify({"message": "No test results found"})
    finally:
        connection.close()
    
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=True)

EDAMAM_APP_ID = '45b0988a'
EDAMAM_APP_KEY = 'cca30a1d41caefa7814f517b1bc1cfa2'


def detect_and_calculate_calories(image_path):
    img = Image.open(image_path)
    results = model(img)
    total_calories = 0
    food_counts = {}
    detected_labels = results.pandas().xyxy[0]['name']

    for label in detected_labels:
        if label in food_counts:
            food_counts[label] += 1
        else:
            food_counts[label] = 1

    for label, count in food_counts.items():
        response = requests.get(
            'https://api.edamam.com/api/nutrition-data',
            params={'app_id': EDAMAM_APP_ID, 'app_key': EDAMAM_APP_KEY, 'ingr': f"{count} {label}"}
        )
        data = response.json()
        if 'calories' in data:
            total_calories += data['calories']
            calories_peritem = data['calories'] / count
            save_calorie_data(label, calories_peritem)


    return total_calories

def save_calorie_data(label, calories):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql_query = """
                INSERT INTO data_kalori (label, kalori)
                VALUES (%s, %s)
            """
            cursor.execute(sql_query, (label, calories))
            connection.commit()
    finally:
        connection.close()

@app.route("/calorie", methods=["POST"])
def calculate_calories():
    image_file = request.files['image']
    image_path = f"/tmp/{image_file.filename}"  
    image_file.save(image_path)

    total_calories = detect_and_calculate_calories(image_path)

    return jsonify({"Total Kalori": total_calories})

if __name__ == "__main__":
    context = ('localhost.crt', 'localhost.key')
    app.run(port=3305, debug=True, ssl_context=context)
