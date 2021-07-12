# all import packages
from flask import Flask, render_template, url_for, redirect, request, session, flash, send_from_directory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import cv2
from werkzeug.utils import secure_filename
import pickle
#from speak import speak,wishme


#extension
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# app config
app = Flask(__name__, static_folder='static', template_folder='template')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


#Covid WB model read
filename = open('CWB/Covidzone.pkl', 'rb')
model = pickle.load(filename)
filename.close()


# all page connection
@app.route('/')
def root():
   #speak("Hi, This is Covid19 Prediction APP")
   return render_template('index.html')

@app.route('/index.html')
def index():
   #speak("Hi, This is Covid19 Prediction APP")
   return render_template('index.html')

@app.route('/contact.html')
def contact():
   #speak("Hi, This is Contact page")
   return render_template('contact.html')

@app.route('/news.html')
def news():
   #speak("Hi, This is News page of covid19")
   return render_template('news.html')

@app.route('/detect.html')
def detect():
   #speak("Hi, This is Covid19 Prediction page")
   return render_template('detect.html')

@app.route('/cwbp.html')
def cwbp():
   return render_template('cwbp.html')

@app.route('/pred', methods=['GET','POST'])
def pred():
   if request.method == "POST":
        
      pos = request.form['pos']
        
      date = request.form["date"]
      Day = int(pd.to_datetime(date, format="%Y-%m-%d").day)
      Month = int(pd.to_datetime(date, format="%Y-%m-%d").month)
      Year = int(pd.to_datetime(date, format="%Y-%m-%d").year)
        
      District = int(request.form['dist'])
      TAC = int(request.form['tac'])
        
      data = np.array([[Day,Month,Year,District,TAC]])
      prediction = model.predict(data)
      # prediction_proba = model.predict_proba(data)[0][1]
        
      return render_template('cwbpshow.html',position=pos,prediction=prediction)#,proba=my_prediction_proba)
   return render_template('cwbp.html')
   


# write the model & store a variable   
mod = load_model('BACP/mymodel.h5')

#predict function for predicting image
def predict(full_name):
   image = cv2.imread(full_name) # read file 
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert the images to greyscale
   image = cv2.resize(image,(224, 224)) #resize images
#    image = cv2.addWeighted (image, 4, cv2.GaussianBlur(image, (0,0), 512/10), -4, 128) #apply Gaussian blur
#    kernel = np.ones((5, 5), np.uint8)
#    image = cv2.erode(image, kernel, iterations=3) #apply Erosion
#    image = cv2.dilate(image, kernel, iterations=3) # apply Dilation
#    image = cv2.Canny(image, 80, 100) #apply Canny edge detection
   image = np.array(image).astype('float32') / 255 #normalize
   image = np.expand_dims(image, axis=0)
   pred = mod.predict(image)
   return pred

@app.route('/upload', methods = ['POST', 'GET'])
def upload():
   if request.method == 'POST':
      file = request.files['file']  
      full_name = os.path.join(UPLOAD_FOLDER, file.filename)
      file.save(full_name)
      
      res = predict(full_name)

      proba = res[0]
      if proba[0] > 0.5:
         pred = str('%.2f' % (proba[0]*100) + '% Covid')
      else:
         pred = str('%.2f' % ((1-proba[0])*100) + '% Healthy')
      
      return render_template('results_chest.html', image_file_name=file.filename, pred=pred)
   return render_template('detect.html') 

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# main calling
if __name__ == '__main__':
   #app.secret_key = ".."
   app.run(port=4040,debug=True)
   #wishme()
