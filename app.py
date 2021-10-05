from flask import Flask
from flask import request
from flask import render_template
from flask_cors import cross_origin
import pickle

#Initializing Flask App
app = Flask(__name__)

#route for homepage
@app.route('/',methods=['GET'])
@cross_origin()
def homepage():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
@cross_origin()
def index():
    if request.method =='POST':
        try:
            CRIM = float(request.form['CRIM'])
            ZN = float(request.form['ZN'])
            INDUS = float(request.form['INDUS'])
            CHAS = float(request.form['CHAS'])
            NOX = float(request.form['NOX'])
            RM = float(request.form['RM'])
            AGE = float(request.form['AGE'])
            DIS = float(request.form['DIS'])
            RAD = float(request.form['RAD'])
            TAX = float(request.form['TAX'])
            PTRATIO = float(request.form['PTRATIO'])
            B = float(request.form['B'])
            LSTAT = float(request.form['LSTAT'])

            filename ='scaler.pickle'
            scaler = pickle.load(open(filename,'rb'))
            x = scaler.fit_transform([[CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT]])

            filename = 'boston prediction.pickle'
            model = pickle.load(open(filename,'rb'))
            prediction = model.predict(x)
            return render_template('results.html',prediction=prediction)
        except Exception as e:
            return'Something is wrong'
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)