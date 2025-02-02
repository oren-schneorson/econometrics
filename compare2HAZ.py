

import os.path
import statsmodels.api as sm
import pandas as pd
from numpy import log
# from datetime import datetime
# from FAME import read_FAME
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('QtAgg')
# ['GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX', 'nbAgg', 'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg',
# 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']

user = input('Please insert username.')

lib_data = '/media/%s/D/data' % user
lib_israel = os.path.join(lib_data, 'Israel')

fname = 'CP_NSA.M.csv'
fpath_NSA = os.path.join(lib_israel, fname)

CP_NSA = pd.read_csv(fpath_NSA, skiprows=7)  # FAME dataset
CP_NSA = CP_NSA.loc[CP_NSA[CP_NSA.columns[-1]].notna()]

CP_NSA.columns = ['date', 'CP_NSA']

CP_NSA.loc[:, 'Pi_NSA'] = CP_NSA.loc[:, 'CP_NSA'].div(CP_NSA.loc[:, 'CP_NSA'].shift(1)).apply(log)*100
CP_NSA.date = pd.to_datetime(CP_NSA.date, dayfirst=True)
CP_NSA = CP_NSA.set_index('date').asfreq('ME', method='bfill')
CP_NSA = CP_NSA.drop(['CP_NSA'], axis=1)


p = 1
d = 0
q = 1

fpath_arima = '/home/%s/Documents/MATLAB/NRC/arima_%d%d%d_il.csv' % (user, p, d, q)

arima = pd.read_csv(fpath_arima)
arima.date = pd.to_datetime(arima.date, dayfirst=False)
arima = arima.set_index('date')

fpath_HAZ = os.path.join(lib_israel, 'HAZ_PI.D.xlsx')
HAZ_PI = pd.read_excel(fpath_HAZ, skiprows=7)
HAZ_PI.columns = ['date'] + list(HAZ_PI.columns[1:])
HAZ_PI = HAZ_PI[['date', 'haz_mad_avg_00m.d']]

HAZ_PI.date = pd.to_datetime(HAZ_PI.date, dayfirst=False)

idx = HAZ_PI.date.dt.day > 20
HAZ_PI = HAZ_PI.loc[idx]
idx = HAZ_PI['haz_mad_avg_00m.d'].notna()
HAZ_PI = HAZ_PI.loc[idx]

HAZ_PI.loc[:, 'year-month'] = HAZ_PI.date.dt.strftime('%Y-%m')
HAZ_PI = HAZ_PI.sort_values(by='date', ascending=True)
HAZ_PI = HAZ_PI.drop_duplicates(subset=['year-month'])
HAZ_PI = HAZ_PI.drop(['year-month'], axis=1)


HAZ_PI = HAZ_PI.set_index('date').asfreq('ME', method='bfill')
HAZ_PI.loc[:, 'date'] = HAZ_PI.index + pd.DateOffset(1)
HAZ_PI.date = HAZ_PI.date + pd.DateOffset(1, months=1)  # this offset is needed
HAZ_PI.loc[:, 'date'] = HAZ_PI.loc[:, 'date'] - pd.DateOffset(1)
HAZ_PI = HAZ_PI.set_index('date')

arima = pd.merge(arima, HAZ_PI, left_on='date', right_on='date')
arima = pd.merge(arima, CP_NSA, left_on='date', right_on='date')
arima.loc[:, 'epsilon_HAZ'] = arima.loc[:, 'Pi_NSA'] - arima.loc[:, 'haz_mad_avg_00m.d']

eps_name = 'epsilon_%d%d%d' % (p, d, q)

arima.epsilon_HAZ.mean()
arima[eps_name].mean()

arima.epsilon_HAZ.autocorr(1)
arima[eps_name].autocorr(1)

print(arima.loc[:, eps_name].corr(arima.loc[:, 'epsilon_HAZ']))

'''
Unexpected inflation:
HAZ is correlated with ARIMA(1,0,1), 62%
HAZ is correlated with ARIMA(0,1,1), 58%

correlation increases a bit if I wait a few more days after the 15th to incorporate other.
'''





