# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 00:44:12 2017

@author: jyots
"""

from glob import glob
import pandas as pd
import numpy as np
from peakdetect import peakdetect

filenames = glob(r'..\OK\*')
filenames_wb = glob(r'..\WB\*')

#==============================================================================
# Related functions
#==============================================================================

def col_11(col_vals):
    y = col_vals
    _max,_min= peakdetect(y,lookahead=75)
    dist = _max[1][0]-_max[0][0] 
    return dist,_max[0][1],_max[1][1]

def col_15(col_vals):
    y = col_vals
    _max,_min= peakdetect(y,lookahead=10)
    dist = _max[1][0]-_max[0][0] 
    return dist,_max[0][1],_max[1][1]

def col_16(col_vals):
    y = col_vals
    _max,_min= peakdetect(y,lookahead=50)
    dist = _max[1][0]-_max[0][0] 
    return dist,_max[0][1],_max[1][1]

def col_18(col_vals):
    y = col_vals
    _max,_min= peakdetect(y,lookahead=100)
    dist = _max[1][0]-_max[0][0] 
    return dist,_max[0][1],_max[1][1]

def col_5(col_vals):
    shift = -col_vals.min()
    new_col = col_vals + shift
    return new_col


def build_features (filename):
    
    
    d = []
    row = []
    columns=[]
    with open(filename,'r') as f:
        for line in f:
            d.append(line)
    columns_ts = d[0].split('\n')[0].split('\t')
    df_ts = pd.DataFrame([],columns=columns_ts)
    
    for i in range(3,25):
        m = d[i].split('\n')[0].split('\t')
        columns.append(m[0])
        
    for i in range(26,len(d)):
        row = (d[i].split('\n')[0].split('\t'))
        df_ts.loc[i-26]= row
    for col in columns_ts:
        df_ts[col]=pd.to_numeric(df_ts[col],errors='ignore')
    df_ts['Time'] = pd.to_datetime(df_ts['Time'])
    
    # Creating new features from the sample data:
    vals_11 = col_11(df_ts[columns_ts[11]])
    vals_15 = col_15(df_ts[columns_ts[15]])
    vals_14 = col_15(df_ts[columns_ts[14]])
    vals_16 = col_16(df_ts[columns_ts[16]])
    vals_18 = col_18(df_ts[columns_ts[18]])
    vals_5 = col_5(df_ts[columns_ts[5]])
    
    #Static data
    row_ = []
    
    for i in range(3,25):
        m = d[i].split('\n')
        n = m[0].split('\t')
        row_.append(n[1])
        
    print (len(row_))
    
    
    print('features built')
    
    row_new = [df_ts[columns_ts[1]].mean(), df_ts[columns_ts[2]].mean(),df_ts[columns_ts[2]].mean(),
           df_ts[columns_ts[10]].mean(),df_ts[columns_ts[9]].mean(),df_ts[columns_ts[8]].mean(),
          df_ts[columns_ts[4]].sum(), vals_5.sum(),vals_5.mean(),df_ts[columns_ts[6]].sum(),
            vals_11[0],vals_11[1],vals_11[2],df_ts[columns_ts[12]].max(),df_ts[columns_ts[12]].min(),
           df_ts[columns_ts[13]].sum(), vals_15[0],vals_15[1],vals_15[2],vals_14[0],vals_14[1],vals_14[2],
           vals_16[0],vals_16[1],vals_16[2],df_ts[columns_ts[17]].sum(),df_ts[columns_ts[19]].sum(),
           vals_18[0],vals_18[1],vals_18[2]
          ]
    print('row_new: ',len(row_new))
    row_new.extend(row_)
    print('combined:', len(row_new))
    print('returning new row')
    return row_new
    
#=============================================================================
# Buidling Datasets
#=============================================================================
# Create Data Set (for OK)


df = []
row_update=[]

for i in range(len(filenames)):

    try:
        row_update= build_features(filenames[i])
        df.append(row_update)
        print('row updates for filenumber: ',i)
    except:
        pass

df =pd.DataFrame(df)
df.to_csv('complete_features_OK.csv',index=False)

# Creating dataset from WB:
df = []
row_update=[]

for i in range(len(filenames_wb)):

    try:
        row_update= build_features(filenames_wb[i])
        df.append(row_update)
        print('row updates for filenumber: ',i)
    except:
        pass

df =pd.DataFrame(df)
df.to_csv('complete_features_WB.csv',index=False)

