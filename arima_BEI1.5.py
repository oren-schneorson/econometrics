
import os.path

import numpy as np
import statsmodels.api as sm
from arch import arch_model
import pandas as pd
from numpy import log, sqrt, array
from numpy import append as np_append
# from datetime import datetime
from FAME import read_FAME
import matplotlib
import matplotlib.pyplot as plt
import shutil
from datetime import datetime

matplotlib.use('QtAgg')
# ['GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX', 'nbAgg', 'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg',
# 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']

user = input('Please insert username.')

subtract_trgt = False

lib_data = '/media/%s/D/data' % user
lib_israel = os.path.join(lib_data, 'Israel')

fname = 'INF_TRGT.D.xlsx'
fpath_data = os.path.join(lib_israel, fname)

# get BEI
# nominal_bond_series = 'TELBOR.M'  # Nov 1999 -->
# nominal_bond, _ = read_FAME(nominal_bond_series, lib_israel, False)

# nominal_bond_series = 'MAKAM_yields.D'
# nominal_bond, _ = read_FAME(nominal_bond_series, lib_israel, False)
# nominal_bond = nominal_bond.asfreq('M', method='ffill')  # to end of month

# TSB_ZRD, _ = read_FAME('TSB_ZRD.D', lib_israel, False)
# TSB_ZRD = TSB_ZRD.asfreq('M', method='ffill')
# TSB_ZRD = TSB_ZRD.shift(periods=1, freq='M')  # shift TSB_ZRD one month forward

TSB_ZRD = pd.read_csv('/media/%s/D/data/Israel/TSB_ZND_15/TSB_ZND_01Y.M_15.csv' % user)
TSB_ZRD.date = pd.to_datetime(TSB_ZRD.date)
TSB_ZRD = TSB_ZRD.set_index('date').asfreq('M', method='ffill')  # shifts two weeks forward
TSB_ZRD = TSB_ZRD.shift(periods=1, freq='M')  # shift one month forward


nominal_bond1 = pd.read_csv('/media/%s/D/data/Israel/MAKAM_yields_15/MAKAM_yields_M01.M_15.csv' % user)
nominal_bond2 = pd.read_csv('/media/%s/D/data/Israel/MAKAM_yields_15/MAKAM_yields_M02.M_15.csv' % user)
nominal_bond = pd.merge(nominal_bond1, nominal_bond2)
nominal_bond.date = pd.to_datetime(nominal_bond.date)
nominal_bond = nominal_bond.set_index('date').asfreq('M', method='ffill')  # shifts two weeks forward
nominal_bond.loc[:, 'nominal_bond_1.5'] = nominal_bond.mean(axis=1)
nominal_bond = nominal_bond.shift(periods=1, freq='M')  # shift one month forward

BEI = pd.merge(nominal_bond[['nominal_bond_1.5']], TSB_ZRD[TSB_ZRD.columns[:1]], left_index=True, right_index=True)
BEI.loc[:, 'BEI'] = (BEI.loc[:, 'nominal_bond_1.5'] - BEI.loc[:, TSB_ZRD.columns[0]])/12


if subtract_trgt:
    INF_TRGT = pd.read_excel(fpath_data, skiprows=7)  # FAME dataset
    INF_TRGT.columns = ['date', ] + list(INF_TRGT.columns[1:])

    cols_trgt = ['INF_MIN_TRGT.D', 'INF_MAX_TRGT.D']
    INF_TRGT.loc[:, 'INF_TRGT'] = INF_TRGT[cols_trgt].mean(axis=1) / 12
    INF_TRGT = INF_TRGT.drop(cols_trgt, axis=1)
    INF_TRGT.columns = ['date', 'INF_TRGT']

    INF_TRGT.date = pd.to_datetime(INF_TRGT.date, dayfirst=False)
    INF_TRGT = INF_TRGT.set_index('date').asfreq('M', method='bfill')
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
data = data.set_index('date').asfreq('M', method='bfill')

data = pd.merge(data, BEI[['BEI']], left_index=True, right_index=True, how='left')
data.loc[:, 'err'] = data.Pi - data.BEI


p = 0
d = 1
q = 1

# for arma of err (TB model)
p_err = 0
d_err = 0
q_err = 1

eps_name = 'epsilon_%d%d%d' % (p, d, q)

fname_arima = 'arima_%d%d%d_il.csv' % (p, d, q)
fpath_arima_NRC = os.path.join('/home/%s/Documents/MATLAB/NRC' % user, fname_arima)
# fpath_arima_Fama1981 = os.path.join('/home/%s/Documents/MATLAB/Fama1981' % user, fname_arima)
fpath_arima_TASE = os.path.join('/home/%s/Documents/MATLAB/TASE' % user, fname_arima)

if d == 0:
    trend = 'c'
else:
    trend = 't'

facsts = []

start_from = 24  # one year of data on

EIAR1 = array([np.nan]*start_from)
EIAR2 = array([np.nan]*start_from)
EIAR3 = array([np.nan]*start_from)

EIRP1 = array([np.nan]*start_from)
EIRP2 = array([np.nan]*start_from)
EIRP3 = array([np.nan]*start_from)

for t in range(start_from, len(data)):

    arima_model = sm.tsa.arima.ARIMA(
        data.Pi.iloc[:t],
        exog=None,
        order=(p, d, q),
        seasonal_order=(0, 0, 0, 0),
        trend=None,
        enforce_stationarity=True,
        enforce_invertibility=True,
        concentrate_scale=False,
        trend_offset=1,
        dates=data.index[:t],
        freq='M',
        missing='none',
        validate_specification=True)

    res = arima_model.fit()  # trained model
    forecast_t = res.forecast(3)

    EIAR1 = np_append(EIAR1, forecast_t.iloc[0])
    EIAR2 = np_append(EIAR2, forecast_t.iloc[1])
    EIAR3 = np_append(EIAR3, forecast_t.iloc[2])

    # TB-model, oos
    arima_model_err = sm.tsa.arima.ARIMA(
        data.err.iloc[:t],
        exog=None,
        order=(p_err, d_err, q_err),
        seasonal_order=(0, 0, 0, 0),
        trend=None,
        enforce_stationarity=True,
        enforce_invertibility=True,
        concentrate_scale=False,
        trend_offset=1,
        dates=data.index[:t],
        freq='M',
        missing='none',
        validate_specification=True)

    res_arima_err = arima_model_err.fit()  # trained model
    forecast_t = res_arima_err.forecast(3)

    EIRP1 = np_append(EIRP1, forecast_t.iloc[0])
    EIRP2 = np_append(EIRP2, forecast_t.iloc[1])
    EIRP3 = np_append(EIRP3, forecast_t.iloc[2])


EIAR2 = np_append(EIAR2[1:], np.nan)
EIAR3 = np_append(EIAR3[2:], [np.nan] * 2)

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

# model = arch_model(data.loc[:, UIAR1_nm].iloc[1:], mean='Zero', vol='GARCH', p=1, q=0)
# model_fit = model.fit()
# model_fit.plot()
# plt.show()
# model_fit.conditional_volatility

# data.loc[:, 'cond_vol'] = np.nan
# data.loc[data.index.isin(data.index[1:]), 'cond_vol'] = model_fit.conditional_volatility


# EIRP1 = np_append(EIRP2[1:], np.nan)
EIRP2 = np_append(EIRP2[1:], np.nan)
EIRP3 = np_append(EIRP2[2:], [np.nan] * 2)

data.loc[:, 'EIRP1'] = EIRP1  # out of sample EIRP (-d_term_spread(1,12)), 1-step ahead
data.loc[:, 'EIRP2'] = EIRP2  # out of sample EIRP (-d_term_spread(1,12)), 2-step ahead
data.loc[:, 'EIRP3'] = EIRP3  # out of sample EIRP (-d_term_spread(1,12)), 3-step ahead

data.loc[:, 'EIRP1'].plot()
data.loc[:, 'EIRP2'].plot()
data.loc[:, 'EIRP3'].plot()

data.loc[:, EIAR1_nm].plot()
data.loc[:, EIAR2_nm].plot()
data.loc[:, EIAR3_nm].plot()


data.loc[:, 'UITB1'] = data.Pi - data.BEI - EIRP1
data.loc[:, 'UITB2'] = data.Pi - data.BEI - EIRP2
data.loc[:, 'UITB3'] = data.Pi - data.BEI - EIRP3

data.loc[:, 'UITB1'].mean()
data.loc[:, 'UITB2'].mean()
data.loc[:, 'UITB3'].mean()
data.loc[:, 'UITB2'].plot()


EIAR2 = np_append(EIAR2[1:], np.nan)
EIAR3 = np_append(EIAR3[2:], [np.nan] * 2)

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

# model = arch_model(data.loc[:, UIAR1_nm].iloc[1:], mean='Zero', vol='GARCH', p=1, q=0)
# model_fit = model.fit()
# model_fit.plot()
# plt.show()
# model_fit.conditional_volatility

# data.loc[:, 'cond_vol'] = np.nan
# data.loc[data.index.isin(data.index[1:]), 'cond_vol'] = model_fit.conditional_volatility


print(sqrt((data.loc[:, 'UITB1'] ** 2).mean()), sqrt((data.loc[:, UIAR1_nm] ** 2).mean()))
print(sqrt((data.loc[:, 'UITB2'] ** 2).mean()), sqrt((data.loc[:, UIAR2_nm] ** 2).mean()))
print(sqrt((data.loc[:, 'UITB3'] ** 2).mean()), sqrt((data.loc[:, UIAR3_nm] ** 2).mean()))


idx = (data.index >= datetime(2010, 1, 31)) & (data.index <= datetime(2024, 3, 31))

sm.graphics.tsa.plot_acf(data.loc[idx, 'UITB1'].values.squeeze(), lags=list(range(1, 37)))
# sm.graphics.tsa.plot_acf(data.loc[data['UITB1'].notna(), 'UITB1'].values[:-1].squeeze(), lags=list(range(1, 37)))
# sm.graphics.tsa.plot_acf(data.loc[data[UIAR3_nm].notna(), UIAR3_nm].values[:-1].squeeze(), lags=list(range(1, 37)))


raise ValueError
cols = ['Pi', 'Pi_hat', eps_name, EIAR1_nm, EIAR2_nm, EIAR3_nm, UIAR1_nm, UIAR2_nm, UIAR3_nm]
data.loc[data[cols].notna().any(axis=1), cols].to_csv(fpath_arima_NRC, index=True)
data.loc[data[cols].notna().any(axis=1), cols].to_csv(fpath_arima_TASE, index=True)

