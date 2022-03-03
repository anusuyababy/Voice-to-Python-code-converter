from flask import Flask, render_template, request
from flask import Flask, render_template
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import operator
import re
import sweetviz 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score 
from sklearn.metrics import accuracy_score
from IPython.display import display
from codegenerator import generate_code
import os
  
# importing pandas module
import pandas as pd

UPLOAD_FOLDER = 'static/'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
        text = request.form['Name']
        #if request.files:
        file = request.files['myfile']
        if file:
            filename = secure_filename(file.filename)
            file.save(app.config['UPLOAD_FOLDER'] + filename)
            
        newfilename = request.form['filename']
        print(newfilename)
        if newfilename:
            filename = newfilename
            
        files = open(app.config['UPLOAD_FOLDER'] + filename,"r")
        
            
        df = pd.read_csv(files)
        return render_template('index.html', result = generate_code(text, df), filename = filename)

                             
if __name__ == "__main__":
    app.run(debug = True)
