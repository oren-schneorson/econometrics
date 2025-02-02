
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

lib_data = '/media/u70o/D/data'
lib_israel = os.path.join(lib_data, 'Israel')

fpath_fear_index = '/media/u70o/D/data/Israel/FEAR_INDEX.D.xlsx'
FEAR_INDEX = pd.read_excel(fpath_fear_index, skiprows=7)  # FAME dataset
FEAR_INDEX.columns = ['date', 'FEAR_INDEX']
FEAR_INDEX.date = pd.to_datetime(FEAR_INDEX.date, dayfirst=False)
FEAR_INDEX = FEAR_INDEX.set_index('date').asfreq('M', method='ffill').shift(periods=1, freq='M')

idx_FEAR_INDEX_censored = FEAR_INDEX.FEAR_INDEX >= np.percentile(FEAR_INDEX.FEAR_INDEX, 75)
FEAR_INDEX.loc[:, 'FEAR_dummy'] = 0
FEAR_INDEX.loc[idx_FEAR_INDEX_censored, 'FEAR_dummy'] = 1

fpath_Rf = '/media/u70o/D/data/Israel/MAKAM_yields/M/MAKAM_yields_M01.M.csv'
nominal_bond = pd.read_csv(fpath_Rf)
nominal_bond.date = pd.to_datetime(nominal_bond.date, dayfirst=False)
# nominal_bond = nominal_bond.set_index('date').asfreq('M', method='ffill')
nominal_bond = nominal_bond.set_index('date').asfreq('M', method='ffill').shift(periods=1, freq='M')
nominal_bond.columns = ['Rf']

fpath_TSB_ZRD = '/media/u70o/D/data/Israel/TSB_ZRD/M/TSB_ZRD_01Y.M.csv'
real_bond = pd.read_csv(fpath_TSB_ZRD)
real_bond.date = pd.to_datetime(real_bond.date, dayfirst=False)
real_bond = real_bond.set_index('date').asfreq('M', method='ffill').shift(periods=1, freq='M')
real_bond.columns = ['rr']

nominal_bond.loc[:, 'dRf'] = nominal_bond['Rf'].diff()
real_bond.loc[:, 'drr'] = real_bond['rr'].diff()

nominal_bond = nominal_bond/12
real_bond = real_bond/12

BEI = pd.merge(nominal_bond, real_bond, left_index=True, right_index=True)
BEI.loc[:, 'BEI'] = BEI.loc[:, 'Rf'] - BEI.loc[:, 'rr']

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

dt, comp_trend_seasonal, comp_Irr = eng.sa_adj(CP_NSA.loc[:, ['CP']].values, 12, nargout=3)
CP_NSA.loc[:, 'CP_SA_sa_adj'] = array(dt, dtype='float64')
CP_NSA.loc[:, 'Pi_SA'] = CP_NSA.loc[:, 'CP_SA_sa_adj'].div(CP_NSA.loc[:, 'CP_SA_sa_adj'].shift(1).astype(float)) - 1
CP_NSA.loc[:, 'Pi_SA'] = CP_NSA.loc[:, 'Pi_SA'] * 100


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


data = pd.merge(data, BEI[['BEI', 'Rf', 'rr', 'dRf', 'drr']], left_index=True, right_index=True, how='left')
data.loc[:, 'BEI_err'] = data.Pi - data.BEI

data = pd.merge(data, FEAR_INDEX[['FEAR_INDEX', 'FEAR_dummy']], left_index=True, right_index=True, how='left')


def train_model(data_, idx_):

    p = 1
    d = 0
    q = 0

    arima_model1 = sm.tsa.arima.ARIMA(
        data_.loc[idx_, 'BEI_err'],
        exog=None,
        order=(p, d, q),
        seasonal_order=(0, 0, 0, 0),
        trend=None,
        enforce_stationarity=True,
        enforce_invertibility=True,
        concentrate_scale=False,
        trend_offset=1,
        dates=data_.index[idx_],
        freq='M',
        missing='none',
        validate_specification=True)

    arima_model1_fit = arima_model1.fit()
    # arima_model1_fit.summary()

    alpha_1 = arima_model1_fit.fittedvalues
    alpha_1_t = arima_model1_fit.forecast()

    BEI_err_post_AR1 = arima_model1_fit.resid

    ols1_model = sm.OLS(BEI_err_post_AR1, sm.add_constant(data_.loc[idx_, ['FEAR_dummy', 'dRf', 'drr']]))
    ols1_model_fit = ols1_model.fit()
    ols1_model_fit.summary()

    alpha_2 = ols1_model_fit.fittedvalues

    data_.loc[idx_, 'ols1_model_fit_resid'] = ols1_model_fit.resid
    z = data_.loc[idx_, 'ols1_model_fit_resid'].diff().values[1:].reshape((idx_.sum()-1, 1))

    _, E_v_given_z = eng.wandering_intercept(z, nargout=2)
    E_v_given_z = array(E_v_given_z, dtype='float64')

    alpha_3 = np.cumsum(E_v_given_z)
    alpha_3_t = alpha_3[-1]
    alpha_3 = np_append([0, 0], alpha_3[:-1])
    alpha_3 = alpha_3-np.mean(alpha_3)

    data_.loc[idx_, 'alpha_1'] = alpha_1
    data_.loc[idx_, 'alpha_2'] = alpha_2
    data_.loc[idx_, 'alpha_3'] = alpha_3

    data_.loc[idx_, 'BEI_clean'] = alpha_1 + alpha_2 + alpha_3 + data_.loc[idx_, 'BEI']

    # ols_model = sm.OLS(data_.loc[idx_, 'Pi']-(alpha_1 + alpha_2 + alpha_3), sm.add_constant(data_.loc[idx_, 'BEI']))
    # ols1_model_fit = ols_model.fit()

    data_.loc[:, 'BEI_err_clean'] = data_.loc[idx_, 'Pi'] - (data_.loc[idx_, 'Pi'] + alpha_1 + alpha_2 + alpha_3)

    p = 1
    d = 0
    q = 0

    arima_model2 = sm.tsa.arima.ARIMA(
        data_.loc[idx_, 'BEI_err_clean'],
        exog=None,
        order=(p, d, q),
        seasonal_order=(0, 0, 0, 0),
        trend=None,
        enforce_stationarity=True,
        enforce_invertibility=True,
        concentrate_scale=False,
        trend_offset=1,
        dates=data_.index[idx_],
        freq='M',
        missing='none',
        validate_specification=True)

    arima_model2_fit = arima_model2.fit()
    arima_model2_fit.summary()

    alpha_4 = arima_model2_fit.fittedvalues
    alpha_4_t = arima_model2_fit.forecast()

    ols2_model = sm.OLS(data_.loc[idx_, 'Pi']-(alpha_1 + alpha_2 + alpha_3-alpha_4), sm.add_constant(data_.loc[idx_, 'BEI']))
    ols2_model_fit = ols2_model.fit()
    ols2_model_fit.summary()

    # UITB = ols2_model_fit.resid  # in sample UITB

    return alpha_1_t, alpha_3_t, alpha_4_t, ols1_model_fit, ols2_model_fit


idx = (data.index >= BEI.index[0]) & (data.index >= FEAR_INDEX.index[0])
data = data.loc[idx]

data.loc[:, 'alpha_t'] = np.nan
data.loc[:, 'forecast'] = np.nan
data.loc[:, 'forecast_error'] = np.nan

for t in range(24, len(data)):
    idx = data.index < data.index[t]
    idx_t = data.index == data.index[t]
    alpha_1_t, alpha_3_t, alpha_4_t, ols1_model_fit, ols2_model_fit = train_model(data, idx)
    alpha_2_t = (data.iloc[t][['FEAR_dummy', 'dRf', 'drr']] * ols1_model_fit.params.iloc[1:]).sum() + ols1_model_fit.params.iloc[0]
    alpha_t = alpha_1_t + alpha_2_t + alpha_3_t + alpha_4_t

    data.loc[idx_t, 'alpha_t'] = alpha_t
    data.loc[idx_t, 'forecast'] = data.iloc[idx_t]['BEI'] - alpha_t
    data.loc[idx_t, 'forecast_error'] = data.iloc[t]['Pi'] - data.iloc[t]['forecast']


data.loc[:, 'forecast_error'].plot()
sqrt((data.loc[:, 'forecast_error'] ** 2).mean())

sm.graphics.tsa.plot_acf(data.loc[data.loc[:, 'forecast_error'].notna(), 'forecast_error'].squeeze(), lags=list(range(1,13)))


data.loc[:, 'alpha_t'].plot()
data.loc[:, 'BEI'].plot()
(data.loc[:, 'Pi']-data.loc[:, 'alpha_t']).plot()
