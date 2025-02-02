
import os.path

import numpy as np
import statsmodels.api as sm
from arch import arch_model
import pandas as pd
from numpy import log, sqrt, array, exp
from numpy import append as np_append
from datetime import datetime
from FAME import read_FAME
import matplotlib
import matplotlib.pyplot as plt
import shutil

import matlab.engine
eng = matlab.engine.start_matlab()

matplotlib.use('QtAgg')
# ['GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX', 'nbAgg', 'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg',
# 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']

user = input('Please insert username.')

lib_data = '/media/u70o/D/data'
lib_israel = os.path.join(lib_data, 'Israel')

fpath_fear_index = '/media/u70o/D/data/Israel/FEAR_INDEX.D.xlsx'
FEAR_INDEX = pd.read_excel(fpath_fear_index, skiprows=7)  # FAME dataset
FEAR_INDEX.columns = ['date', 'FEAR_INDEX']
FEAR_INDEX.date = pd.to_datetime(FEAR_INDEX.date, dayfirst=False)
FEAR_INDEX = FEAR_INDEX.set_index('date').asfreq('M', method='ffill').shift(periods=1, freq='M')

idx_FEAR_INDEX_censored = FEAR_INDEX.FEAR_INDEX >= np.percentile(FEAR_INDEX.FEAR_INDEX, 99)
FEAR_INDEX.loc[:, 'FEAR_dummy'] = 0
FEAR_INDEX.loc[idx_FEAR_INDEX_censored, 'FEAR_dummy'] = 1


fpath_inf_trgt = '/media/u70o/D/data/Israel/INF_TRGT.M.xlsx'
INF_TRGT = pd.read_excel(fpath_inf_trgt, skiprows=7)  # FAME dataset
cols_trgt = ['INF_MIN_TRGT.M', 'INF_MAX_TRGT.M']
INF_TRGT.columns = ['date', ] + cols_trgt

INF_TRGT.loc[:, 'INF_TRGT'] = INF_TRGT[cols_trgt].mean(axis=1) / 12
INF_TRGT = INF_TRGT.drop(cols_trgt, axis=1)
INF_TRGT.columns = ['date', 'INF_TRGT']

INF_TRGT.date = pd.to_datetime(INF_TRGT.date, dayfirst=False)
INF_TRGT = INF_TRGT.set_index('date').asfreq('M', method='ffill')
# INF_TRGT = INF_TRGT.set_index('date').asfreq('M', method='ffill').shift(periods=1, freq='M')
# INF_TRGT.loc[:, 'date'] = INF_TRGT.index.strftime('%d/%m/%Y')
# INF_TRGT = INF_TRGT.reset_index(drop=True)


fpath_Rf = '/media/u70o/D/data/Israel/MAKAM_yields/M/MAKAM_yields_M01.M.csv'
nominal_bond = pd.read_csv(fpath_Rf)
nominal_bond.date = pd.to_datetime(nominal_bond.date, dayfirst=False)
# nominal_bond = nominal_bond.set_index('date').asfreq('M', method='ffill')
nominal_bond = nominal_bond.set_index('date').asfreq('M', method='ffill').shift(periods=1, freq='M')

# nominal_bond_series = 'MAKAM_yields.D'
# nominal_bond, _ = read_FAME(nominal_bond_series, lib_israel, False)
# nominal_bond = nominal_bond.asfreq('M', method='ffill')  # to end of month

fpath_TSB_ZRD = '/media/u70o/D/data/Israel/TSB_ZRD/M/TSB_ZRD_01Y.M.csv'
TSB_ZRD = pd.read_csv(fpath_TSB_ZRD)
TSB_ZRD.date = pd.to_datetime(TSB_ZRD.date, dayfirst=False)
# TSB_ZRD = TSB_ZRD.set_index('date').asfreq('M', method='ffill')
TSB_ZRD = TSB_ZRD.set_index('date').asfreq('M', method='ffill').shift(periods=1, freq='M')

# TSB_ZRD, _ = read_FAME('TSB_ZRD.D', lib_israel, False)
# TSB_ZRD = TSB_ZRD.asfreq('M', method='ffill')
# TSB_ZRD = TSB_ZRD.shift(periods=1, freq='M')  # shift TSB_ZRD one month forward

BEI = pd.merge(nominal_bond[nominal_bond.columns[:1]], TSB_ZRD[TSB_ZRD.columns[:1]], left_index=True, right_index=True)
BEI.loc[:, 'BEI'] = (BEI.loc[:, nominal_bond.columns[0]] - BEI.loc[:, TSB_ZRD.columns[0]])/12

fname = 'CP_NSA.M.csv'
fpath_CP_NSA = os.path.join(lib_israel, fname)

CP_NSA = pd.read_csv(fpath_CP_NSA, skiprows=7)  # FAME dataset
CP_NSA = CP_NSA.loc[CP_NSA['CP'].notna()]
CP_NSA.columns = ['date', 'CP']
CP_NSA.date = pd.to_datetime(CP_NSA.date, dayfirst=True)
CP_NSA = CP_NSA.set_index('date')
CP_NSA = CP_NSA.asfreq('M')
CP_NSA.loc[:, 'log(CP)'] = CP_NSA['CP'].apply(log)
CP_NSA.loc[:, 'dlog(CP)'] = CP_NSA['log(CP)'].diff()*100
CP_NSA.loc[:, 'Pi_NSA'] = CP_NSA.loc[:, 'CP'].div(CP_NSA.loc[:, 'CP'].shift(1)) - 1
CP_NSA.loc[:, 'Pi_NSA'] = CP_NSA.loc[:, 'Pi_NSA'] * 100

first_date = datetime(1986, 1, 31)
idx = CP_NSA.index >= first_date
CP_NSA = CP_NSA.loc[idx]

# CP_NSA.loc[idx, 'Pi'].plot()
# CP_NSA.loc[idx, 'dlog(CP)'].plot()

dt, comp_trend_seasonal, comp_Irr = eng.sa_adj(CP_NSA.loc[:, ['CP']].values, 12, nargout=3)
CP_NSA.loc[:, 'CP_SA_sa_adj'] = array(dt, dtype='float64')
CP_NSA.loc[:, 'Pi_SA'] = CP_NSA.loc[:, 'CP_SA_sa_adj'].div(CP_NSA.loc[:, 'CP_SA_sa_adj'].shift(1).astype(float)) - 1
CP_NSA.loc[:, 'Pi_SA'] = CP_NSA.loc[:, 'Pi_SA'] * 100

# CP_NSA.loc[idx, 'Pi_NSA'].plot()
# CP_NSA.loc[idx, 'Pi_SA'].plot()
# CP_NSA.loc[idx, 'CP'].apply(log).plot()
# CP_NSA.loc[idx, 'CP_SA_sa_adj'].apply(log).plot()

fname = 'CP_SA.M.csv'
fpath_CP_SA = os.path.join(lib_israel, fname)

data = pd.read_csv(fpath_CP_SA, skiprows=7)  # FAME dataset
data = data.loc[data[data.columns[-1]].notna()]

data.columns = ['date', 'CP_SA']

# data.loc[:, 'Pi'] = data.loc[:, 'CP_SA'].div(data.loc[:, 'CP_SA'].shift(1)).apply(log)*100
data.loc[:, 'Pi'] = data.loc[:, 'CP_SA'].div(data.loc[:, 'CP_SA'].shift(1)) - 1
data.loc[:, 'Pi'] = data.loc[:, 'Pi'] * 100


data.date = pd.to_datetime(data.date, dayfirst=True)
data = data.set_index('date').asfreq('M')

data = pd.merge(data, CP_NSA, how='right', left_index=True, right_index=True)

idx = data.Pi.isna()
data.loc[idx, 'Pi'] = data.loc[idx, 'Pi_SA']

data = pd.merge(data, INF_TRGT, left_index=True, right_index=True)
data.loc[:, 'Pi_dev_trgt'] = data.loc[:, 'Pi'] - data.loc[:, 'INF_TRGT']  # deviation from target (detrend)

data.loc[:, 'Pi_dev_trgt'].mean()
data.loc[:, 'Pi_dev_trgt'].plot()

data = pd.merge(data, BEI[['BEI']], left_index=True, right_index=True, how='left')
data.loc[:, 'err'] = data.Pi - data.BEI

data = pd.merge(data, FEAR_INDEX[['FEAR_INDEX', 'FEAR_dummy']], left_index=True, right_index=True, how='left')

idx = (data.index > BEI.index[0]) & (data.index > FEAR_INDEX.index[0])

# data['err'].plot()
# data['Pi'].mean()
# plt.show()

p = 1
q = 1

arch_model_Pi_dev = arch_model(data.loc[idx, 'Pi_dev_trgt'], mean='zero', vol='GARCH', p=p, q=q)
arch_model_Pi_dev_fit = arch_model_Pi_dev.fit()
data.loc[idx, 'cond_vol'] = arch_model_Pi_dev_fit.conditional_volatility
arch_model_Pi_dev_fit.plot()


p = 1
d = 0
q = 0

data.loc[:, 'Pi'].iloc[61:62]
data.loc[:, 'err'].iloc[61:62]

data.loc[idx, 'err'].plot()

arima_model = sm.tsa.arima.ARIMA(
    data.loc[idx, 'err'],
    exog=None,
    order=(p, d, q),
    seasonal_order=(0, 0, 0, 0),
    trend=None,
    enforce_stationarity=True,
    enforce_invertibility=True,
    concentrate_scale=False,
    trend_offset=1,
    dates=data.index[idx],
    freq='M',
    missing='none',
    validate_specification=True)

arima_model_fit = arima_model.fit()
arima_model_fit.summary()

data.loc[idx, 'err_resid_ar1'] = arima_model_fit.resid
data.loc[idx, 'err_resid_ar1'].mean()
data.loc[idx, 'err_resid_ar1'].plot()

idx = data.index > FEAR_INDEX.index[0]

model_ols = sm.OLS(data.loc[idx, 'err_resid_ar1'], sm.add_constant(data.loc[idx, ['FEAR_INDEX', 'FEAR_dummy']]))
model_ols_fit = model_ols.fit()
model_ols_fit.summary()
model_ols_fit.resid.plot()















q = 1
p = 1
arch_model_err = arch_model(data.loc[idx, 'err_resid_ar1'], mean='zero', vol='GARCH', p=p, q=q)
arch_model_err_fit = arch_model_err.fit()
arch_model_err_fit.summary()

arch_model_err_fit.plot()





model = arch_model(data.loc[idx, 'err'], vol='GARCH', p=p, q=q)
model_fit = model.fit()
data.loc[idx, 'cond_vol'] = model_fit.conditional_volatility
model_fit.plot()

model_fit.resid.plot()

model_ols = sm.OLS(data.loc[idx, 'err'], sm.add_constant(data.loc[idx, 'cond_vol']))
model_ols_fit = model_ols.fit()
model_ols_fit.summary()
model_ols_fit.resid.plot()

data['err'].plot()

df = []
for p in range(1, 15):
    for q in range(0, 15):

        model = arch_model(data.loc[idx, 'Pi_dev_trgt'], vol='GARCH', p=p, q=q)
        model_fit = model.fit()
        row = (p, q, model_fit.aic, model_fit.bic, model_fit)
        df.append(row)


df = pd.DataFrame(df, columns=['p', 'q', 'aic', 'bic', 'model_fit'])
df = df.sort_values(by='aic', )
df

df.iloc[0]['model_fit'].plot()
# model_fit.plot()
# plt.show()
# model_fit.conditional_volatility

# data.loc[idx, 'err'].plot()
# plt.show()

