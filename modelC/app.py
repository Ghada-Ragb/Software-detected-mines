from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='template')

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template("Mine2.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final = np.array(int_features)
    print(final)
    prediction = model.predict([final])[0]

    if prediction == 1:
        return render_template('Mine2.html',
                               pred='You are in Danger.\nProbability of Mine occuring is {}'.format(prediction))
    else:
        return render_template('Mine2.html',
                               pred='Your are safe.\n Probability of Mine occuring is {}'.format(prediction))


if __name__ == '__main__':
    app.run(debug=True)
