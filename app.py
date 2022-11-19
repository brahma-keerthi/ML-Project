import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn import preprocessing

app = Flask(__name__)
model = pickle.load(open('model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #print(request.form)
    int_features = [float(x) if x != '' else 0 for x in request.form.values()]
    print("I F =",int_features)
    final = []
    final.append(int_features)
    final_features = preprocessing.normalize(final)

    prediction = model.predict(final_features)
    output = np.round(prediction, 2)

    return render_template('pred.html', ev1=output[0][0], ev2=output[0][1],ev3=output[0][2],ev4=output[0][3],ev5=output[0][4],ev6=output[0][5],ev7=output[0][6])

if __name__ == "__main__":
    app.run(debug=True, port=4996)