import numpy as np 
from flask import Flask, request, render_template, redirect, flash, url_for
from werkzeug.utils import secure_filename
from main import getPrediction
import os

UPLOAD_FOLDER = '/home/bertam/CHALLENGE/deployment/static'

app= Flask(__name__, static_folder= "static")

app.secret_key = 'una_clau_secreta_molt_secreta'

app.config['UPLOAD_FOLDER']= UPLOAD_FOLDER

@app.route('/') #home
def home():
    return render_template('home.html')

@app.route('/predict/', methods=['POST', 'GET']) 
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('NO file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename) #Protegir info pacients
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            prediction_class, probability_gc, probability_occ = getPrediction(filename)
            flash(prediction_class)
            flash(probability_gc)
            flash(probability_occ)
            flash(url_for('static', filename=f'{filename}'))  # original image
            flash(url_for('static', filename=f'images/overlay_{filename}'))  # overlay image
            flash(url_for('static', filename=f'images/occlusion_{filename}'))  # overlay image
            return redirect('/prediction/')
    else:
        return render_template('predict.html')

@app.route('/prediction/')
def prediction():
    return render_template('prediction.html')


#Això de baix serveix per dir: si estic executant el codi en aquest document, fes app.run()
if __name__ == "__main__":
    app.run(port=5002, debug=True)