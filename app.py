from flask import Flask, render_template,request, jsonify
import requests
from weather import getweather
import pickle
import json
app=Flask(__name__)


with open('flood_model1.pkl','rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/calculate',methods=['POST'])
def calculate():
    response = request.get_json()
    city=response['city']
    weather=getweather(city)
    prediction = model.predict_proba(weather)[0][1]
    risk_percent = round(prediction * 100, 2)
    return jsonify({'result':prediction})

@app.route('/predict')
def predict():
    return render_template('predict.html')

if __name__=='__main__':
    app.run(debug=True)