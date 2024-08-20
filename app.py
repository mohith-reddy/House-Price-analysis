import flask
import pandas as pd
from joblib import dump, load


with open(f'housepriceprediction1.joblib', 'rb') as f:
    model = load(f)


app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('main.html'))

    if flask.request.method == 'POST':
        rooms = flask.request.form['rooms']
        bathroom = flask.request.form['bathroom']
        landsize = flask.request.form['landsize']
        lattitude = flask.request.form['lattitude']
        longtitude = flask.request.form['longtitude']
        distance = flask.request.form['distance']
        car = flask.request.form['car']
        
        buildingarea = flask.request.form['buildingarea']
        yearbuilt = flask.request.form['yearbuilt']
        propertycount = flask.request.form['propertycount']

        input_variables = pd.DataFrame([[rooms, bathroom, landsize, lattitude, longtitude, distance, car, landsize, buildingarea, yearbuilt]],
                                       columns=['rooms', 'bathroom', 'landsize', 'lattitude', 'longtitude',
                                                'distance', 'car', 'buildingarea', 'yearbuilt','propertycount'],
                                       dtype='float',
                                       index=['input'])

        predictions = model.predict(input_variables)[0]
        print(predictions)

        return flask.render_template('main.html', original_input={'Rooms': rooms, 'Bathroom': bathroom, 'Landsize': landsize, 'Lattitude': lattitude, 'Longtitude': longtitude, 'Distance': distance, 'Car': car, 'BuildingArea': buildingarea, 'YearBuilt': yearbuilt, 'Propertycount': propertycount},
                                     result=predictions)


if __name__ == '__main__':
    app.run(debug=True)
