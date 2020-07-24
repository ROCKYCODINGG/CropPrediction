import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json

app = Flask(__name__)
model = pickle.load(open('venv\\Include\\model.pkl', 'rb'))


@app.route('/predict', methods=['POST'])
def predict():
    data = json.dumps(request.form)
    ko = json.loads(data)

    #     return ko['Temprature']
    kl = [float(ko['Rainfall']), float(ko['Temprature']), float(ko['N']), float(ko['P']),
          float(ko['K']), float(ko['Ph'])]
    a = ['Bajra', 'Banana', 'Barley', 'Bean', 'Black pepper', 'Blackgram',
         'Bottle Gourd', 'Brinjal', 'Cabbage', 'Cardamom', 'Carrot',
         'Castor seed', 'Cauliflower', 'Chillies', 'Colocosia', 'Coriander',
         'Cotton', 'Cowpea', 'Drum Stick', 'Garlic', 'Ginger', 'Gram',
         'Grapes', 'Groundnut', 'Guar seed', 'Horse-gram', 'Jowar', 'Jute',
         'Khesari', 'Lady Finger', 'Lentil', 'Linseed', 'Maize', 'Mesta',
         'Moong(Green Gram)', 'Moth', 'Onion', 'Orange', 'Papaya',
         'Peas & beans (Pulses)', 'Pineapple', 'Potato', 'Raddish', 'Ragi',
         'Rice', 'Safflower', 'Sannhamp', 'Sesamum', 'Soyabean',
         'Sugarcane', 'Sunflower', 'Sweet potato', 'Tapioca', 'Tomato',
         'Turmeric', 'Urad', 'Varagu', 'Wheat']

    #     b=int(DT.predict([kl]))
    prediction = model.predict([kl])
    output = prediction[0]
    k = a[output]
    return str(k)


if __name__ == "__main__":
    app.run()