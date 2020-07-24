import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model

from sklearn.metrics import accuracy_score
# low_memory=False
a = pd.read_csv("C:\\Users\\nicem\\PycharmProjects\\CropPrediction\\venv\\Include\\regression.csv",error_bad_lines=False, index_col=False, dtype='unicode')

a['P'] = a['P'].replace(['17', '15', '3', '8', '7', '24', '22', '21', '8+D163954', "9"],
                        [17, 15, 3, 8, 7, 24, 22, 21, 8, 9])

label = LabelEncoder()

a['Crop'] = label.fit_transform(a['Crop'])

x = a.drop('Crop', axis=1)  # indepen
y = a['Crop']  # Depned

x = np.array(x)
y = np.array(y)

# y_test

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

from sklearn import tree

DT = tree.DecisionTreeClassifier()
DT.fit(x_train, y_train)  # dt

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

b = int(DT.predict([[452.98507, 23.34327, 10, 15, 230, 6.7]]))
print(a[b])

pickle.dump(DT, open('model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('C:\\Users\\nicem\\PycharmProjects\\CropPrediction\\venv\\Include\\model.pkl', 'rb'))
print(model.predict([[452.98507, 23.34327, 10, 15, 230, 6.7]]))  # pickle file model898/>/