# -*- coding: utf-8 -*-
"""
Created on Wed May 11 17:23:24 2022

@author: hmasmoudi
"""
import h5py 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
        
print("1")
#from pmdarima.arima import auto_arima
#from statsmodels.tsa.arima.model import ARIMA
#import statsmodels.api as sm
from datetime import datetime


def mape_(y,y_hat):
    mapee = np.mean(np.abs(y-y_hat)/y)*100
    return mapee

def process_data(y,date):
    df= pd.DataFrame()
    df['ds'] = pd.to_datetime(date)
    df['y']=y
    return df

datadir= '/home/daf/opal/analytics/xmi/sys_tests/sys_data/elect_data/'

train = h5py.File(datadir + "train.hdf5","r")
test = h5py.File(datadir + "test.hdf5","r")
date_train =  pd.read_csv(datadir + "train_times.txt",header=None)
date_test = pd.read_csv(datadir + "test_times.txt",header=None)
# Fetch hours index 
time_train = np.array(train.get("unhot_X_calendar_hour"))
time_test = np.array(test.get("unhot_X_calendar_hour"))
# Target values range from 1-49999 train and 50000-62197 test
y_out = np.array(train.get("y_out"))
Ty_out = np.array(test.get("y_out"))
# Sift sales for each country, for loop 
location = np.array(train.get("unhot_X_location"))
location_test = np.array(test.get("unhot_X_location"))
locations_index = np.unique(location)
print("7")
# Computing auto_arima on each location 

dates_train = [datetime.strptime(h[0], "%Y-%m-%d-%H") for h in date_train.values]
dates_test = [datetime.strptime(h[0], "%Y-%m-%d-%H") for h in date_test.values]

from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
 
if True:
    models = []
    for loc in [3]:
        print(f"\nTraining prophet country: {loc}\n")
        sift = location == loc
        sift_test = location_test ==loc
        y_out_loc = y_out[sift]
        daits = np.array(dates_train)[sift.flatten()]
        
        Ty_out_loc = Ty_out[sift_test]        
        Tdaits = np.array(dates_test)[sift_test.flatten()]
        tdays = (daits.max()-daits.min()).days
        yy = np.append(y_out_loc,Ty_out_loc)
        xx = np.append(daits,Tdaits)
        
        data_df =  process_data(yy,xx) 
                                
        m = Prophet(daily_seasonality=True)
        m.fit(data_df) # Fit it with all the data - just to get the model initialised. 
        # There are 168 hours in a week - we want a randomish sample so take every 151nd hour = 81 forecasts. 
        df_cv = cross_validation(m,horizon="1 days",period="51 hours",initial=str(tdays) + ' days',parallel="processes")
        # Take only every 24th sample from this as this is a 24 hour ahead forecast. 
        sloc = np.arange(23,df_cv.shape[0],24)
        y = np.array(df_cv.y[sloc])
        y_hat = np.array(df_cv.yhat[sloc])
        
        mape1 = mape_(y,y_hat)
        bias = np.mean(y-y_hat)
        rmse = np.sqrt(((y - y_hat) ** 2).mean())
        print(mape1)
        print(rmse)
        print(bias)

# loc =1
#7.992175160349239
#1148.4362316805205
#-203.50145745411606
 