# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 11:30:27 2022

@author: kennedy
"""

# importing flask
from flask import Flask, render_template, request
from flask import Flask, render_template
import numpy as np
import pandas as pd
import os
import operator
import re
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
from sklearn.metrics import accuracy_score
from IPython.display import display
from codegenerator import generate_code
import os
  
# importing pandas module
import pandas as pd
  
  
app = Flask(__name__)
  
  
# reading the data in the csv file
df = pd.read_csv('salary.csv')
df.to_csv('salary.csv', index=None)
  
@app.route('/')
def index():
   return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
       df = pd.read_csv('salary.csv')
       text = request.form['Name']
       
       
       return render_template("index.html",result = generate_code(text))
  

  
if __name__ == "__main__":
    app.run(debug = True)
