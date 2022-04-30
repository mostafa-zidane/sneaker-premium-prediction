import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd
import datetime

app = Flask(__name__)

prmodel = pickle.load(open('regressor.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    column_names = ['Unnamed: 0', 'Retail Price', 'Shoe Size', '1', '2015', '2016', '2017',
                    '2018', '2Pt0', '350', '90', '97', 'Abloh', 'Adidas', 'Af100', 'Air',
                    'All', 'Beluga', 'Black', 'Blazer', 'Blue', 'Boost', 'Butter',
                    'Chicago', 'Copper', 'Core', 'Cream', 'Desert', 'Elemental', 'Eve',
                    'Fly', 'Flyknit', 'Force', 'Frozen', 'Green', 'Grey', 'Grim', 'Hallows',
                    'High', 'Hyperdunk', 'Jordan', 'Low', 'Max', 'Menta', 'Mercurial',
                    'Mid', 'Moonrock', 'Nike', 'Offwhite', 'Orange', 'Ore', 'Oxford',
                    'Pink', 'Pirate', 'Presto', 'Queen', 'React', 'Reaper', 'Red',
                    'Reflective', 'Retro', 'Rose', 'Semi', 'Sesame', 'Silver', 'Static',
                    'Tan', 'Tint', 'Total', 'Turtledove', 'University', 'V2', 'Vapormax',
                    'Virgil', 'Volt', 'White', 'Wolf', 'Yeezy', 'Yellow', 'Zebra', 'Zoom',
                    'Alabama', 'Release_Date']
    print(column_names)
    dff = pd.DataFrame([np.zeros(len(column_names))], columns=column_names)
    features = list(request.form.values())
    print(features)
    df = dff.copy()
    try:
        sneaker = str(features[0])
        sneaker = sneaker.replace("_", " ")
        snk = sneaker.split(' ')
        for k in snk:
            s = str(k.lower().capitalize())
            try:
                df[[s]]
                df[s] = 1
            except:
                continue
    except:
        print('error')
        return render_template('index.html', prediction_text='ERROR: INVALID NAME')
    try:
        size = float(features[1])
        df['Shoe Size'] = size
    except:
        return render_template('index.html', prediction_text='ERROR: INVALID SHOE SIZE')
    try:
        rdyear = int(features[2])
        rdmonth = int(features[3])
        rdday = int(features[4])
        date = datetime.datetime(rdyear, rdmonth, rdday)
        date0 = (date - datetime.datetime(2000, 1, 1)).days
        df['Release_Date'] = date0
    except:
        output = 'NA, invalid date'
        return render_template('index.html', prediction_text='ERROR: INVALID DATE')
    try:
        price = int(features[5])
        df['Retail Price'] = price
    except:
        print('price error')
        return render_template('index.html', prediction_text='ERROR: INVALID PRICE')
    try:
        pred = prmodel.predict(df)
        output = round(pred[0], 4)
    except:
        return render_template('index.html', prediction_text='ERROR: INVALID ENTRIES')

    return render_template('index.html', prediction_text='PREMIUM PREDICTION: {}'.format(output))


if __name__ == "__main__":
    app.static_folder = 'static'
    app.run()
