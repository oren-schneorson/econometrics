
import os.path

import numpy as np
import statsmodels.api as sm
from arch import arch_model
import pandas as pd
from numpy import log, sqrt, array, exp
from numpy import append as np_append
from datetime import datetime
# from FAME import read_FAME
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

fname = 'INF_TRGT.D.xlsx'
fpath_data = os.path.join(lib_israel, fname)

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

data['Pi'].plot()
# data['Pi'].mean()
# plt.show()

p = 1
q = 0

# model = arch_model(data.loc[:, 'Pi'].iloc[1:], mean='Zero', vol='GARCH', p=p, q=q)
model = arch_model(data.loc[:, 'Pi'].iloc[1:], vol='GARCH', p=p, q=q)
model_fit = model.fit()
model_fit.plot()
model_fit.conditional_volatility



