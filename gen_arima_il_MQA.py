import os.path

import numpy as np
import statsmodels.api as sm
import pandas as pd
from numpy import log, sqrt, array
from numpy import append as np_append
# from datetime import datetime
# from FAME import read_FAME
import matplotlib
import matplotlib.pyplot as plt
import shutil
from load_CP_NSA import *

matplotlib.use('QtAgg')
# ['GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX', 'nbAgg', 'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg',
# 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']

user = input('Please insert username.')

lib_data = '/media/%s/D/data' % user
lib_israel = os.path.join(lib_data, 'Israel')

fname = 'INF_TRGT.D.xlsx'
fpath_data = os.path.join(lib_israel, fname)

subtract_trgt = True

freq = 'Q'

if subtract_trgt:
    INF_TRGT = pd.read_excel(fpath_data, skiprows=7)  # FAME dataset
    INF_TRGT.columns = ['date', ] + list(INF_TRGT.columns[1:])

    cols_trgt = ['INF_MIN_TRGT.D', 'INF_MAX_TRGT.D']
    if freq == 'M':
        INF_TRGT.loc[:, 'INF_TRGT'] = INF_TRGT[cols_trgt].mean(axis=1) / 12
    elif freq == 'Q':
        INF_TRGT.loc[:, 'INF_TRGT'] = INF_TRGT[cols_trgt].mean(axis=1) / 4
    elif freq == 'A':
        INF_TRGT.loc[:, 'INF_TRGT'] = INF_TRGT[cols_trgt].mean(axis=1) / 1

    INF_TRGT = INF_TRGT.drop(cols_trgt, axis=1)
    INF_TRGT.columns = ['date', 'INF_TRGT']

    INF_TRGT.date = pd.to_datetime(INF_TRGT.date, dayfirst=False)
    INF_TRGT = INF_TRGT.set_index('date').asfreq('M', method='ffill')
    # INF_TRGT.loc[:, 'date'] = INF_TRGT.index.strftime('%d/%m/%Y')
    # INF_TRGT = INF_TRGT.reset_index(drop=True)

fname = 'CP_SA.M.csv'
fpath_data = os.path.join(lib_israel, fname)

data = pd.read_csv(fpath_data, skiprows=7)  # FAME dataset
data = data.loc[data[data.columns[-1]].notna()]

data.columns = ['date', 'CP_SA']

data.date = pd.to_datetime(data.date, dayfirst=True)
data = data.set_index('date').asfreq('M', method='ffill')

data = pd.merge(data, CP_NSA[['CP_SA_sa_adj']], how='right', left_index=True, right_index=True)
data.loc[:, 'CP_SA_sa_adj'] = data.loc[:, 'CP_SA_sa_adj'] / data.loc[
    data.index == datetime(1996, 1, 31), 'CP_SA_sa_adj'].values
data.loc[:, 'CP_SA_sa_adj'] = data.loc[:, 'CP_SA_sa_adj'] * data.loc[
    data.index == datetime(1996, 1, 31), 'CP_SA'].values

idx = data.index.year < 1996
data.loc[idx, 'CP_SA'] = data.loc[idx, 'CP_SA_sa_adj']

if freq == 'M':
    # data.loc[:, 'Pi'] = data.loc[:, 'CP_SA'].div(data.loc[:, 'CP_SA'].shift(1)).apply(log)*100
    data.loc[:, 'Pi'] = data.loc[:, 'CP_SA'].div(data.loc[:, 'CP_SA'].shift(1)) - 1
    data.loc[:, 'Pi'] = data.loc[:, 'Pi'] * 100
elif freq == 'Q':
    data = data.asfreq('Q')
    # data.loc[:, 'Pi'] = data.loc[:, 'CP_SA'].div(data.loc[:, 'CP_SA'].shift(1)).apply(log)*100
    data.loc[:, 'Pi'] = data.loc[:, 'CP_SA'].div(data.loc[:, 'CP_SA'].shift(1)) - 1
    data.loc[:, 'Pi'] = data.loc[:, 'Pi'] * 100

elif freq == 'A':
    data = CP_NSA.copy()
    data = data.asfreq('A')
    data.loc[:, 'Pi'] = data.loc[:, 'CP'].div(data.loc[:, 'CP'].shift(1)) - 1
    data.loc[:, 'Pi'] = data.loc[:, 'Pi'] * 100


if subtract_trgt:
    data = pd.merge(data, INF_TRGT, how='left', left_index=True, right_index=True)
    data.loc[:, 'Pi'] = data.loc[:, 'Pi'] - data.loc[:, 'INF_TRGT']  # deviation from target (detrend)

idx = data['Pi'].notna()
# idx = (data.index >= datetime(2001,10,1)) & (data.index < datetime(2024,2,1))   # for in-sample
data = data.loc[idx]


p = 1
d = 0
q = 1

fname_arima = 'arima_%d%d%d_%s_il.csv' % (p, d, q, freq)
fpath_arima_NRC = os.path.join('/home/%s/Documents/MATLAB/NRC' % user, fname_arima)
# fpath_arima_Fama1981 = os.path.join('/home/%s/Documents/MATLAB/Fama1981' % user, fname_arima)
fpath_arima_TASE = os.path.join('/home/%s/Documents/MATLAB/TASE' % user,  fname_arima)
fpath_arima_UIAR = os.path.join(lib_israel, 'UIAR.xlsx')

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
    freq=freq,
    missing='none',
    validate_specification=True)

res = arima_model.fit()

with open('arima_latex.txt', 'w') as fp:
    fp.write(res.summary().as_latex())

# res.predict()
# res.predict(dynamic=0)
# res.predict(dynamic=1)
# res.predict() + res.forecasts_error[0] - data.Pi

if freq == 'M':
    start_forecasting = 119
elif freq == 'Q':
    start_forecasting = 39
elif freq == 'A':
    start_forecasting = 9

EIAR1 = array([np.nan] * (start_forecasting - 0))
EIAR2 = array([np.nan] * (start_forecasting - 0))
EIAR3 = array([np.nan] * (start_forecasting - 0))

for t in range(start_forecasting, len(data)):
    arima_model_t = sm.tsa.arima.ARIMA(
        data.iloc[:t]['Pi'],
        exog=None,
        order=(p, d, q),
        seasonal_order=(0, 0, 0, 0),
        trend=None,
        enforce_stationarity=True,
        enforce_invertibility=True,
        concentrate_scale=False,
        trend_offset=1,
        dates=data.index[:t],
        freq=freq,
        missing='none',
        validate_specification=True)

    res_t = arima_model_t.fit()
    forecast_t = res.predict(start=t + 1, end=t + 3, information_set='predicted').values

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
data.loc[:, eps_name] = res.forecasts_error.transpose()  # in sample prediction errors
# to see that these are in-sample:
# data.Pi-data.INF_TRGT-(res.fittedvalues+res.forecasts_error[0])  # OR
# data.Pi-(res.fittedvalues+res.forecasts_error[0])  # if didn't return the INF_TRGT

print(sqrt(data.loc[:, eps_name].apply(lambda x: x ** 2).mean()))
print(sqrt(data.loc[:, UIAR1_nm].apply(lambda x: x ** 2).mean()))  # out of sample, 1-step ahead
print(sqrt(data.loc[:, UIAR2_nm].apply(lambda x: x ** 2).mean()))  # # out of sample, 2-step ahead
print(sqrt(data.loc[:, UIAR3_nm].apply(lambda x: x ** 2).mean()))  # # out of sample, 2-step ahead

if subtract_trgt:
    data.loc[:, 'Pi'] = data.loc[:, 'Pi'] + data.loc[:, 'INF_TRGT']
    data.loc[:, EIAR1_nm] = EIAR1 + data.loc[:, 'INF_TRGT']
    data.loc[:, EIAR2_nm] = EIAR2 + data.loc[:, 'INF_TRGT']
    data.loc[:, EIAR3_nm] = EIAR3 + data.loc[:, 'INF_TRGT']

data.loc[:, 'Pi_hat'] = data.loc[:, 'Pi'] - data.loc[:, eps_name]  # in-sample fitted values
# data = data.drop(['INF_TRGT'], axis=1)


# data.loc[:, 'ma'] = data.loc[:, eps_name].rolling(60).mean()
# data.loc[:, 'ma'] = data.loc[:, UIAR1_nm].rolling(60).mean()
data.loc[:, 'ma'] = data.loc[:, UIAR2_nm].rolling(60).mean()

'''
for lag in list(range(1, 37)):
    # print(lag, data.loc[:, eps_name].iloc[1:].autocorr(lag))
    # print(lag, data.loc[:, UIAR1_nm].iloc[1:].autocorr(lag))
    print(lag, data.loc[:, UIAR2_nm].iloc[1:].autocorr(lag))
'''

data.loc[:, 'ma'].iloc[1:].autocorr(1)

data.loc[:, eps_name].iloc[1:].mean()
sqrt(data.loc[:, eps_name].iloc[1:].apply(lambda x: x ** 2).mean())  # .34 deviation from trgt

cols = ['Pi', 'Pi_hat', eps_name, EIAR1_nm, EIAR2_nm, EIAR3_nm, UIAR1_nm, UIAR2_nm, UIAR3_nm]
data.loc[data[cols].notna().any(axis=1), cols].to_csv(fpath_arima_NRC, index=True)
data.loc[data[cols].notna().any(axis=1), cols].to_csv(fpath_arima_TASE, index=True)
data.loc[data[[UIAR1_nm]].notna().any(axis=1), [UIAR1_nm]].to_excel(fpath_arima_UIAR, index=True)
# data.loc[:, 'ma'].plot()
# data[[eps_name, 'ma']].plot()
# plt.show(block=False)


'''
Subtracting INF_TRGT from Pi, and then attempting to predict Pi it results in a UIAR with 0 autocorrelation.
This should be very close to the TB model, where the TBill is doing the job of INF_TRGT to some extent.
'''


data[['Pi', EIAR1_nm]].plot()