
import os.path

import numpy as np
import statsmodels.api as sm
from arch import arch_model
import pandas as pd
from numpy import log, sqrt, array
from numpy import append as np_append
# from datetime import datetime
# from FAME import read_FAME
import matplotlib
import matplotlib.pyplot as plt
import shutil

# import matlab.engine
# matlab.engine.start_matlab


matplotlib.use('QtAgg')
# ['GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX', 'nbAgg', 'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg',
# 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']

lib_data = '/media/u70o/D/data'
lib_israel = os.path.join(lib_data, 'Israel')

fname = 'INF_TRGT.D.xlsx'
fpath_data = os.path.join(lib_israel, fname)

subtract_trgt = False

if subtract_trgt:
    INF_TRGT = pd.read_excel(fpath_data, skiprows=7)  # FAME dataset
    INF_TRGT.columns = ['date', ] + list(INF_TRGT.columns[1:])

    cols_trgt = ['INF_MIN_TRGT.D', 'INF_MAX_TRGT.D']
    INF_TRGT.loc[:, 'INF_TRGT'] = INF_TRGT[cols_trgt].mean(axis=1) / 12
    INF_TRGT = INF_TRGT.drop(cols_trgt, axis=1)
    INF_TRGT.columns = ['date', 'INF_TRGT']

    INF_TRGT.date = pd.to_datetime(INF_TRGT.date, dayfirst=False)
    INF_TRGT = INF_TRGT.set_index('date').asfreq('M', method='ffill')
    INF_TRGT.loc[:, 'date'] = INF_TRGT.index.strftime('%d/%m/%Y')
    INF_TRGT = INF_TRGT.reset_index(drop=True)

fname = 'CP_SA.M.csv'
fpath_data = os.path.join(lib_israel, fname)

data = pd.read_csv(fpath_data, skiprows=7)  # FAME dataset
data = data.loc[data[data.columns[-1]].notna()]

data.columns = ['date', 'CP_SA']

# data.loc[:, 'Pi'] = data.loc[:, 'CP_SA'].div(data.loc[:, 'CP_SA'].shift(1)).apply(log)*100
data.loc[:, 'Pi'] = data.loc[:, 'CP_SA'].div(data.loc[:, 'CP_SA'].shift(1)) - 1
data.loc[:, 'Pi'] = data.loc[:, 'Pi'] * 100

if subtract_trgt:
    data = pd.merge(data, INF_TRGT)
    data.loc[:, 'Pi'] = data.loc[:, 'Pi'] - data.loc[:, 'INF_TRGT']  # deviation from target (detrend)

data.date = pd.to_datetime(data.date, dayfirst=True)
data = data.set_index('date').asfreq('M', method='ffill')

# data['Pi'].plot()
# data['Pi'].mean()
# plt.show()

p = 0
d = 1
q = 1

fname_arima = 'arima_%d%d%d_il.csv' % (p, d, q)
fpath_arima_NRC = os.path.join('/home/u70o/Documents/MATLAB/NRC', fname_arima)
# fpath_arima_Fama1981 = os.path.join('/home/u70o/Documents/MATLAB/Fama1981', fname_arima)
fpath_arima_TASE = os.path.join('/home/u70o/Documents/MATLAB/TASE', fname_arima)

if d == 0:
    trend = 'c'
else:
    trend = 't'

arima_model = sm.tsa.arima.ARIMA(
    data.Pi,
    exog=None,
    order=(p, d, q),
    seasonal_order=(0, 0, 0, 0),
    trend=None,
    enforce_stationarity=True,
    enforce_invertibility=True,
    concentrate_scale=False,
    trend_offset=1,
    dates=data.index,
    freq='M',
    missing='none',
    validate_specification=True)

res = arima_model.fit()

with open('arima_latex.txt', 'w') as fp:
    fp.write(res.summary().as_latex())

# res.predict()
# res.predict(dynamic=0)
# res.predict(dynamic=1)
res.predict() + res.forecasts_error[0] - data.Pi

EIAR1 = array([])
EIAR2 = array([])
EIAR3 = array([])

for t in range(1, len(data) + 1):
    forecast_t = res.predict(start=t, end=t + 2, information_set='predicted').values
    EIAR1 = np_append(EIAR1, forecast_t[0])
    EIAR2 = np_append(EIAR2, forecast_t[1])
    EIAR3 = np_append(EIAR3, forecast_t[2])

EIAR1_nm = 'EIAR1_%d%d%d' % (p, d, q)
EIAR2_nm = 'EIAR2_%d%d%d' % (p, d, q)
EIAR3_nm = 'EIAR3_%d%d%d' % (p, d, q)
UIAR1_nm = 'UIAR1_%d%d%d' % (p, d, q)
UIAR2_nm = 'UIAR2_%d%d%d' % (p, d, q)
UIAR3_nm = 'UIAR3_%d%d%d' % (p, d, q)



data.loc[:, EIAR1_nm] = EIAR1
data.loc[:, EIAR2_nm] = EIAR2
data.loc[:, EIAR3_nm] = EIAR3
data.loc[:, UIAR1_nm] = data.Pi - data.loc[:, EIAR1_nm]  # out of sample, 1-step ahead
data.loc[:, UIAR2_nm] = data.Pi - data.loc[:, EIAR2_nm]  # out of sample, 2-step ahead
data.loc[:, UIAR3_nm] = data.Pi - data.loc[:, EIAR3_nm]  # out of sample, 3-step ahead

# x1 = data.loc[:, UIAR1_nm].iloc[1:].apply(lambda x: x**2)
# x2 = data.loc[:, UIAR2_nm].iloc[1:].apply(lambda x: x**2)
# x3 = data.loc[:, UIAR3_nm].iloc[1:].apply(lambda x: x**2)
#
# sm.tsa.stattools.acf(x1)
# sm.tsa.stattools.acf(x2)
#
# sm.graphics.tsa.plot_acf(x1.values.squeeze(), lags=list(range(1,13)))
# sm.graphics.tsa.plot_acf(x2.values.squeeze(), lags=list(range(1,13)))
# plt.show()

eps_name = 'epsilon_%d%d%d' % (p, d, q)
data.loc[:, eps_name] = res.forecasts_error.transpose()  # in sample predictions
# to see that these are in-sample:
# data.Pi-data.INF_TRGT-(res.fittedvalues+res.forecasts_error[0])  # OR
# data.Pi-(res.fittedvalues+res.forecasts_error[0])  # if didn't return the INF_TRGT

print(sqrt(data.loc[:, eps_name].apply(lambda x: x ** 2).mean()))
print(sqrt(data.loc[:, UIAR1_nm].apply(lambda x: x ** 2).mean()))  # out of sample, 1-step ahead
print(sqrt(data.loc[:, UIAR2_nm].apply(lambda x: x ** 2).mean()))  # # out of sample, 2-step ahead
print(sqrt(data.loc[:, UIAR3_nm].apply(lambda x: x ** 2).mean()))  # # out of sample, 2-step ahead

if subtract_trgt:
    data.loc[:, 'Pi'] = data.loc[:, 'Pi'] + data.loc[:, 'INF_TRGT']

data.loc[:, 'Pi_hat'] = data.loc[:, 'Pi'] - data.loc[:, eps_name]  # in-sample fitted values
# data = data.drop(['INF_TRGT'], axis=1)


model = arch_model(data.loc[:, UIAR1_nm].iloc[1:], mean='Zero', vol='GARCH', p=1, q=1)
model_fit = model.fit()
model_fit.plot()
model_fit.conditional_volatility

plt.show()

data.loc[:, UIAR1_nm].plot()
plt.show()
