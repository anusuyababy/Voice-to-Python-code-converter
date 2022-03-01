# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 11:47:21 2022

@author: kennedy
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import operator
import re
import dtale
from IPython.display import display
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

df=pd.read_csv("salary.csv")
def generate_code(query):
    if operator.contains(query, "import library"):
        print('import pandas as pd') 
        print('import numpy as np') 
        print('import os') 
        print('import plotly.express as px') 
        print('import matplotlib.pyplot as plt') 
        print('import seaborn as sns')
        return
    elif operator.contains(query, "libraries"):
        print('import pandas as pd') 
        print('import numpy as np') 
        print('import os') 
        print('import plotly.express as px') 
        print('import matplotlib.pyplot as plt') 
        print('import seaborn as sns')
        return
    elif operator.contains(query, "head"):
        s = df.head().to_html()
        return s
    elif operator.contains(query, "shape"):
        s = df.shape
        return s
    elif operator.contains(query, "null"):
        s = df.isnull().sum()
        d = s.to_frame()
        f = d.to_html()
        return f
    elif operator.contains(query, "null values"):
        s = df.isnull().sum()
        d = s.to_frame()
        f = d.to_html()
        return f
    elif operator.contains(query, "nullvalues"):
        s = df.isnull().sum()
        d = s.to_frame()
        f = d.to_html()
        return f
    elif operator.contains(query, "correlation"):
        d = df.corr().to_html()
        return d
    elif operator.contains(query, "list the columns"):
        s = df.columns
        d = s.to_frame()
        q = d.reset_index(drop=True)
        f = q.to_html()
        return f
    elif operator.contains(query, "column names"):
        s = df.columns
        d = s.to_frame()
        q = d.reset_index(drop=True)
        f = q.to_html()
        return f
    elif operator.contains(query, "column name"):
        s = df.columns
        d = s.to_frame()
        q = d.reset_index(drop=True)
        f = q.to_html()
        return f
    elif operator.contains(query, "names"):
        s = df.columns
        d = s.to_frame()
        q = d.reset_index(drop=True)
        f = q.to_html()
        return f
    elif operator.contains(query, "description"):
        d = df.describe().to_html() 
        return d
    elif operator.contains(query, "row"):
        row = int(''.join(filter(str.isdigit, query)))
        s = display(df.loc[row:row])
        return s
    elif operator.contains(query, "numerical feature"):
        featnum=[feature for feature in df.columns if df[feature].dtype !='O']
        s = df[featnum].head(5).to_html()
        return s
    elif operator.contains(query, "numerical features"):
        featnum=[feature for feature in df.columns if df[feature].dtype !='O']
        s = df[featnum].head(5).to_html()
        return s
    elif operator.contains(query, "numerical variable"):
        featnum=[feature for feature in df.columns if df[feature].dtype !='O']
        s = df[featnum].head(5).to_html()
        return s
    elif operator.contains(query, "numerical variables"):
        featnum=[feature for feature in df.columns if df[feature].dtype !='O']
        s = df[featnum].head(5).to_html()
        return s
    elif operator.contains(query, "categorical feature"):
        featcat=[feature for feature in df.columns if df[feature].dtype =='O']
        s = df[featcat].head(5).to_html()
        return s
    elif operator.contains(query, "categorical features"):
        featcat=[feature for feature in df.columns if df[feature].dtype =='O']
        s = df[featcat].head(5).to_html()
        return s
    elif operator.contains(query, "categorical variable"):
        featcat=[feature for feature in df.columns if df[feature].dtype =='O']
        s = df[featcat].head(5).to_html()
        return s
    elif operator.contains(query, "categorical variables"):
        featcat=[feature for feature in df.columns if df[feature].dtype =='O']
        s = df[featcat].head(5).to_html()
        return s
    elif operator.contains(query, "missing"):
        df.replace('?', np.NaN, inplace=True)
        # seperating categorical feature
        feat_cat=[feature for feature in df.columns if df[feature].isnull().sum()>0 and df[feature].dtype =='O']
        # seperating numerical feature
        feat_num=[feature for feature in df.columns if df[feature].isnull().sum()>0 and df[feature].dtype !='O']
        # taking median for numerical features
        s = df[feat_num].median()
        df[feat_num]=df[feat_num].fillna(s)
        # filling the missing values with "missing" for categorical features
        df[feat_cat]=df[feat_cat].fillna('Missing')
        s = df.head().to_html()
        return s
    elif operator.contains(query, "fill the missing"):
        df.replace('?', np.NaN, inplace=True)
        # seperating categorical feature
        feat_cat=[feature for feature in df.columns if df[feature].isnull().sum()>0 and df[feature].dtype =='O']
        # seperating numerical feature
        feat_num=[feature for feature in df.columns if df[feature].isnull().sum()>0 and df[feature].dtype !='O']
        # taking median for numerical features
        s = df[feat_num].median()
        df[feat_num]=df[feat_num].fillna(s)
        # filling the missing values with "missing" for categorical features
        df[feat_cat]=df[feat_cat].fillna('Missing')
        s = df.head().to_html()
        return s
    elif operator.contains(query, "clean the data"):
        df.replace('?', np.NaN, inplace=True)
        # seperating categorical feature
        feat_cat=[feature for feature in df.columns if df[feature].isnull().sum()>0 and df[feature].dtype =='O']
        # seperating numerical feature
        feat_num=[feature for feature in df.columns if df[feature].isnull().sum()>0 and df[feature].dtype !='O']
        # taking median for numerical features
        s = df[feat_num].median()
        df[feat_num]=df[feat_num].fillna(s)
        # filling the missing values with "missing" for categorical features
        df[feat_cat]=df[feat_cat].fillna('Missing')
        s = df.head().to_html()
        return s
    elif operator.contains(query, "clean"):
        df.replace('?', np.NaN, inplace=True)
        # seperating categorical feature
        feat_cat=[feature for feature in df.columns if df[feature].isnull().sum()>0 and df[feature].dtype =='O']
        # seperating numerical feature
        feat_num=[feature for feature in df.columns if df[feature].isnull().sum()>0 and df[feature].dtype !='O']
        # taking median for numerical features
        s = df[feat_num].median()
        df[feat_num]=df[feat_num].fillna(s)
        # filling the missing values with "missing" for categorical features
        df[feat_cat]=df[feat_cat].fillna('Missing')
        s = df.head().to_html()
        return s
    elif operator.contains(query, "data preprocessing"):
        df.replace('?', np.NaN, inplace=True)
        # seperating categorical feature
        feat_cat=[feature for feature in df.columns if df[feature].isnull().sum()>0 and df[feature].dtype =='O']
        # seperating numerical feature
        feat_num=[feature for feature in df.columns if df[feature].isnull().sum()>0 and df[feature].dtype !='O']
        # taking median for numerical features
        s = df[feat_num].median()
        df[feat_num]=df[feat_num].fillna(s)
        # filling the missing values with "missing" for categorical features
        df[feat_cat]=df[feat_cat].fillna('Missing')
        s = df.head().to_html()
        return s
    elif operator.contains(query, "label encoding"):
        le = LabelEncoder()
        objList = df.select_dtypes(include = "object").columns
        print (objList)
        for feat in objList:
            df[feat] = le.fit_transform(df[feat].astype(str))
        s = df.head(5).to_html()
        return s
    elif operator.contains(query, "feature engineering"):
        le = LabelEncoder()
        objList = df.select_dtypes(include = "object").columns
        print (objList)
        for feat in objList:
            df[feat] = le.fit_transform(df[feat].astype(str))
        s = df.head(5).to_html()
        return s
    elif operator.contains(query, "label encoder"):
        le = LabelEncoder()
        objList = df.select_dtypes(include = "object").columns
        print (objList)
        for feat in objList:
            df[feat] = le.fit_transform(df[feat].astype(str))
        s = df.head(5).to_html()
        return s
    elif operator.contains(query, "encode"):
        le = LabelEncoder()
        objList = df.select_dtypes(include = "object").columns
        print (objList)
        for feat in objList:
            df[feat] = le.fit_transform(df[feat].astype(str))
        s = df.head(5).to_html()
        return s
    elif operator.contains(query, "top"):
        n = int(''.join(filter(str.isdigit, query)))
        s = df.head(n).to_html()
        return s
    elif operator.contains(query, "bottom"):
        n = int(''.join(filter(str.isdigit, query)))
        s = df.tail(n).to_html()
        return s
    elif operator.contains(query, "split the data"):
        n = int(''.join(filter(str.isdigit, query)))
        s = int(str(n)[-2:])
        d = s/100
        x = df.iloc[:,:-1]
        y = df.iloc[: , -1:]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=d, random_state=0)
        d = X_train.shape, X_test.shape
        return d
    elif operator.contains(query, "divide the data"):
        n = int(''.join(filter(str.isdigit, query)))
        s = int(str(n)[-2:])
        d = s/100
        x = df.iloc[:,:-1]
        y = df.iloc[: , -1:]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=d, random_state=0)
        d = X_train.shape, X_test.shape
        return d
    elif operator.contains(query, "ratio"):
        n = int(''.join(filter(str.isdigit, query)))
        s = int(str(n)[-2:])
        d = s/100
        x = df.iloc[:,:-1]
        y = df.iloc[: , -1:]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=d, random_state=0)
        d = X_train.shape, X_test.shape
        return d
    elif operator.contains(query, "into"):
        n = int(''.join(filter(str.isdigit, query)))
        s = int(str(n)[-2:])
        d = s/100
        x = df.iloc[:,:-1]
        y = df.iloc[: , -1:]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=d, random_state=0)
        d = X_train.shape, X_test.shape
        return d
    elif operator.contains(query, "to"):
        n = int(''.join(filter(str.isdigit, query)))
        s = int(str(n)[-2:])
        d = s/100
        x = df.iloc[:,:-1]
        y = df.iloc[: , -1:]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=d, random_state=0)
        d = X_train.shape, X_test.shape
        return d
    elif operator.contains(query, "build  the model"):
        x = df.iloc[:,:-1]
        y = df.iloc[: , -1:]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
        # Gaussian naive bayes
        clf = GaussianNB()
        clf.fit(x_train, y_train)
        NBscore = clf.score(x_train, y_train)
        y_pred=clf.predict(x_test)
        p = accuracy_score(y_test,y_pred)
        # random forest
        clf1 = RandomForestClassifier(n_estimators=20)
        clf1.fit(x_train, y_train)
        rfscore = clf1.score(x_train, y_train)
        y_pred1=clf1.predict(x_test)
        q = accuracy_score(y_test,y_pred1)
        # KNN Classifier
        clf2 = KNeighborsClassifier()
        clf2.fit(x_train, y_train)
        knnscore = clf2.score(x_train, y_train)
        y_pred2=clf2.predict(x_test)
        r = accuracy_score(y_test,y_pred2)
        # Decision tree classifier
        clf3 = DecisionTreeClassifier()
        clf3.fit(x_train, y_train)
        dtscore = clf3.score(x_train, y_train)
        y_pred3=clf3.predict(x_test)
        s = accuracy_score(y_test,y_pred3)
        # SVM Classifier
        clf4 = svm.SVC()
        clf4.fit(x_train, y_train)
        svmscore = clf4.score(x_train, y_train)
        y_pred4=clf4.predict(x_test)
        t = accuracy_score(y_test,y_pred4)
        # adaboost classifier
        clf5 = AdaBoostClassifier()
        clf5.fit(x_train, y_train)
        adascore = clf5.score(x_train, y_train)
        y_pred5=clf5.predict(x_test)
        u = accuracy_score(y_test,y_pred5)
        # gradient boosting
        clf6 = GradientBoostingClassifier()
        clf6.fit(x_train, y_train)
        gradientscore = clf6.score(x_train, y_train)
        y_pred6=clf6.predict(x_test)
        v = accuracy_score(y_test,y_pred6)
        report = pd.DataFrame()
        report['Algorithm'] = ['Gaussian Naive bayes', 'Random Forest', 'KNN', 'Decission Tree', 'SVM', 'Adaboost', 'Gradient boosting']
        report['Training Accuracy'] = [NBscore, rfscore, knnscore, dtscore, svmscore, adascore, gradientscore]
        report['Test Accuracy'] = [p, q, r, s, t, u, v]
        z = report.to_html()
        return z
    elif operator.contains(query, "model building"):
        x = df.iloc[:,:-1]
        y = df.iloc[: , -1:]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
        # Gaussian naive bayes
        clf = GaussianNB()
        clf.fit(x_train, y_train)
        NBscore = clf.score(x_train, y_train)
        y_pred=clf.predict(x_test)
        p = accuracy_score(y_test,y_pred)
        # random forest
        clf1 = RandomForestClassifier(n_estimators=20)
        clf1.fit(x_train, y_train)
        rfscore = clf1.score(x_train, y_train)
        y_pred1=clf1.predict(x_test)
        q = accuracy_score(y_test,y_pred1)
        # KNN Classifier
        clf2 = KNeighborsClassifier()
        clf2.fit(x_train, y_train)
        knnscore = clf2.score(x_train, y_train)
        y_pred2=clf2.predict(x_test)
        r = accuracy_score(y_test,y_pred2)
        # Decision tree classifier
        clf3 = DecisionTreeClassifier()
        clf3.fit(x_train, y_train)
        dtscore = clf3.score(x_train, y_train)
        y_pred3=clf3.predict(x_test)
        s = accuracy_score(y_test,y_pred3)
        # SVM Classifier
        clf4 = svm.SVC()
        clf4.fit(x_train, y_train)
        svmscore = clf4.score(x_train, y_train)
        y_pred4=clf4.predict(x_test)
        t = accuracy_score(y_test,y_pred4)
        # adaboost classifier
        clf5 = AdaBoostClassifier()
        clf5.fit(x_train, y_train)
        adascore = clf5.score(x_train, y_train)
        y_pred5=clf5.predict(x_test)
        u = accuracy_score(y_test,y_pred5)
        # gradient boosting
        clf6 = GradientBoostingClassifier()
        clf6.fit(x_train, y_train)
        gradientscore = clf6.score(x_train, y_train)
        y_pred6=clf6.predict(x_test)
        v = accuracy_score(y_test,y_pred6)
        report = pd.DataFrame()
        report['Algorithm'] = ['Gaussian Naive bayes', 'Random Forest', 'KNN', 'Decission Tree', 'SVM', 'Adaboost', 'Gradient boosting']
        report['Training Accuracy'] = [NBscore, rfscore, knnscore, dtscore, svmscore, adascore, gradientscore]
        report['Test Accuracy'] = [p, q, r, s, t, u, v]
        z = report.to_html()
        return z
    elif operator.contains(query, "model training"):
        x = df.iloc[:,:-1]
        y = df.iloc[: , -1:]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
        # Gaussian naive bayes
        clf = GaussianNB()
        clf.fit(x_train, y_train)
        NBscore = clf.score(x_train, y_train)
        y_pred=clf.predict(x_test)
        p = accuracy_score(y_test,y_pred)
        # random forest
        clf1 = RandomForestClassifier(n_estimators=20)
        clf1.fit(x_train, y_train)
        rfscore = clf1.score(x_train, y_train)
        y_pred1=clf1.predict(x_test)
        q = accuracy_score(y_test,y_pred1)
        # KNN Classifier
        clf2 = KNeighborsClassifier()
        clf2.fit(x_train, y_train)
        knnscore = clf2.score(x_train, y_train)
        y_pred2=clf2.predict(x_test)
        r = accuracy_score(y_test,y_pred2)
        # Decision tree classifier
        clf3 = DecisionTreeClassifier()
        clf3.fit(x_train, y_train)
        dtscore = clf3.score(x_train, y_train)
        y_pred3=clf3.predict(x_test)
        s = accuracy_score(y_test,y_pred3)
        # SVM Classifier
        clf4 = svm.SVC()
        clf4.fit(x_train, y_train)
        svmscore = clf4.score(x_train, y_train)
        y_pred4=clf4.predict(x_test)
        t = accuracy_score(y_test,y_pred4)
        # adaboost classifier
        clf5 = AdaBoostClassifier()
        clf5.fit(x_train, y_train)
        adascore = clf5.score(x_train, y_train)
        y_pred5=clf5.predict(x_test)
        u = accuracy_score(y_test,y_pred5)
        # gradient boosting
        clf6 = GradientBoostingClassifier()
        clf6.fit(x_train, y_train)
        gradientscore = clf6.score(x_train, y_train)
        y_pred6=clf6.predict(x_test)
        v = accuracy_score(y_test,y_pred6)
        report = pd.DataFrame()
        report['Algorithm'] = ['Gaussian Naive bayes', 'Random Forest', 'KNN', 'Decission Tree', 'SVM', 'Adaboost', 'Gradient boosting']
        report['Training Accuracy'] = [NBscore, rfscore, knnscore, dtscore, svmscore, adascore, gradientscore]
        report['Test Accuracy'] = [p, q, r, s, t, u, v]
        z = report.to_html()
        return z
    elif operator.contains(query, "train the model"):
        x = df.iloc[:,:-1]
        y = df.iloc[: , -1:]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
        # Gaussian naive bayes
        clf = GaussianNB()
        clf.fit(x_train, y_train)
        NBscore = clf.score(x_train, y_train)
        y_pred=clf.predict(x_test)
        p = accuracy_score(y_test,y_pred)
        # random forest
        clf1 = RandomForestClassifier(n_estimators=20)
        clf1.fit(x_train, y_train)
        rfscore = clf1.score(x_train, y_train)
        y_pred1=clf1.predict(x_test)
        q = accuracy_score(y_test,y_pred1)
        # KNN Classifier
        clf2 = KNeighborsClassifier()
        clf2.fit(x_train, y_train)
        knnscore = clf2.score(x_train, y_train)
        y_pred2=clf2.predict(x_test)
        r = accuracy_score(y_test,y_pred2)
        # Decision tree classifier
        clf3 = DecisionTreeClassifier()
        clf3.fit(x_train, y_train)
        dtscore = clf3.score(x_train, y_train)
        y_pred3=clf3.predict(x_test)
        s = accuracy_score(y_test,y_pred3)
        # SVM Classifier
        clf4 = svm.SVC()
        clf4.fit(x_train, y_train)
        svmscore = clf4.score(x_train, y_train)
        y_pred4=clf4.predict(x_test)
        t = accuracy_score(y_test,y_pred4)
        # adaboost classifier
        clf5 = AdaBoostClassifier()
        clf5.fit(x_train, y_train)
        adascore = clf5.score(x_train, y_train)
        y_pred5=clf5.predict(x_test)
        u = accuracy_score(y_test,y_pred5)
        # gradient boosting
        clf6 = GradientBoostingClassifier()
        clf6.fit(x_train, y_train)
        gradientscore = clf6.score(x_train, y_train)
        y_pred6=clf6.predict(x_test)
        v = accuracy_score(y_test,y_pred6)
        report = pd.DataFrame()
        report['Algorithm'] = ['Gaussian Naive bayes', 'Random Forest', 'KNN', 'Decission Tree', 'SVM', 'Adaboost', 'Gradient boosting']
        report['Training Accuracy'] = [NBscore, rfscore, knnscore, dtscore, svmscore, adascore, gradientscore]
        report['Test Accuracy'] = [p, q, r, s, t, u, v]
        z = report.to_html()
        return z