
import os.path
import sys
import statsmodels.api as sm
import pandas as pd
from FAME import *
from numpy import log
from datetime import datetime

user = input('Please insert username.')

lib_data = '/media/%s/D/data' % user
lib_israel = os.path.join(lib_data, 'Israel')

fpath_arima_101 = '/home/%s/Documents/MATLAB/NRC/legacy/arima_101_il.csv' % user
fpath_arima_011 = '/home/%s/Documents/MATLAB/NRC/legacy/arima_011_il.csv' % user

arima_101 = pd.read_csv(fpath_arima_101)
arima_011 = pd.read_csv(fpath_arima_011)

arima_101.date = pd.to_datetime(arima_101.date, dayfirst=True)
arima_011.date = pd.to_datetime(arima_011.date, dayfirst=True)

arima_011.date = arima_011.date + pd.DateOffset(1)
arima_011.date = arima_011.date - pd.DateOffset(1, months=1)
arima_011.date = arima_011.date - pd.DateOffset(1)

arima_101.date = arima_101.date + pd.DateOffset(1)
arima_101.date = arima_101.date - pd.DateOffset(1, months=1)
arima_101.date = arima_101.date - pd.DateOffset(1)


fname = 'CP_SA.M.xlsx'

data, metadata = read_FAME('CP_SA.M', lib_israel, dayfirst_=True)
data.columns = ['CP_SA']
data = data.reset_index()
data.loc[:, 'Pi'] = data.loc[:, 'CP_SA'].div(data.loc[:, 'CP_SA'].shift(1)).apply(log)*100

data.loc[:, 'Pi'].mean()*12
data = data.loc[data.date < datetime(2023, 8, 1)]
data = data.loc[~data.Pi.isna()]
p = 0
d = 1
q = 1

endog = data.loc[:, data.columns[0]]
arima_model = sm.tsa.arima.ARIMA(
    arima_011.Pi,
    exog=None,
    order=(p, d, q),
    seasonal_order=(0, 0, 0, 0),
    trend='t',
    enforce_stationarity=True,
    enforce_invertibility=True,
    concentrate_scale=False,
    trend_offset=1,
    dates=arima_011.date,
    freq=None,
    missing='none',
    validate_specification=True)

res = arima_model.fit()

print(res.summary())
res.forecasts_error.shape
data.shape

arima_011.loc[:, 'eps'] = res.forecasts_error.transpose()
arima_011.loc[:, 'eps'].iloc[1:]

arima_011.loc[:, 'epsilon_011'].iloc[1:].autocorr(1)
arima_011.loc[:, 'eps'].iloc[1:].autocorr(1)

arima_011.loc[:, 'epsilon_011'].iloc[1:].mean()
arima_011.loc[:, 'eps'].iloc[1:].mean()

arima_011.loc[:, 'epsilon_011'].iloc[1:].corr(arima_011.loc[:, 'eps'].iloc[1:])

'''
Results are very close.
'''