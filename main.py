# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 00:49:28 2017

@author: jyots
"""

import pandas as pd
import numpy as np
from sklearn import metrics
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import scipy.stats as ss
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

Std = StandardScaler()

# Reading previously created dataset
ok = pd.read_csv('complete_features_OK.csv',index_col=False)
columns = ok.columns
for col in columns:
    ok[col]=pd.to_numeric(ok[col],errors='ignore')
col_disc = [col_n for col_n in range(len(columns)) if ok[columns[col_n]].dtype=='str' or ok[columns[col_n]].dtype=='object']
ok.drop(['30','34'],axis=1,inplace=True)
ok_normalize = Std.fit_transform(ok)
ok_normalize = pd.DataFrame(ok_normalize)
ok_normalize['label']=0

wb = pd.read_csv('complete_features_WB.csv',index_col=False)
wb.drop(['30','34'],axis=1,inplace=True)
wb_normalize = Std.fit_transform(wb)
wb_normalize = pd.DataFrame(wb_normalize)
wb_normalize['label'] = 1

df = pd.concat([ok_normalize,wb_normalize],ignore_index=True)
columns = df.columns

#Feature Selection
def get_R_continuous(X, Y, m=5):
    

    Z = np.zeros((len(X), m))

    temp_y = np.array(Y)

    lr = LinearRegression()

    R_list = []

    for i in range(0, m):

        Z[:, i] = X ** (i + 1)

        lr.fit(Z[:, 0: i + 1], temp_y)

        R_list.append(round(lr.score(Z[:, 0: i + 1], Y), 4))

    R_max = max(R_list)

    return R_max

result_cont = pd.DataFrame(index=df[columns[:-1]].columns, columns=['correlation'])

for col in df[columns[:-1]].columns:

    index = df.index
    result_cont['correlation'].loc[col] = get_R_continuous(df[col][index], df['label'][index], 3)

result_cont.sort_values(['correlation'], axis=0, ascending=False,
                            inplace=True)

index_top = result_cont[:15]

#=============================================================================
#  Split to train and test set
#=============================================================================
X_train, X_test, y_train, y_test = train_test_split(df[index_top],df['label'], test_size=.1)

#=============================================================================
#  Learning Model
#==============================================================================
clf_svm = SVC(kernel='rbf',C=1000,gamma=0.01)
clf_svm.fit(X_train,y_train)
y_pred = clf_svm.predict(X_test)
print ('Accuracy: ', metrics.accuracy_score(y_pred=y_pred,y_true=y_test))
print ('Precision: ', metrics.precision_score(y_pred=y_pred,y_true=y_test))
print ('Recall score: ', metrics.recall_score(y_pred=y_pred,y_true=y_test))
print(metrics.confusion_matrix(y_test,y_pred))
