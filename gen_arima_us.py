
import os.path
import statsmodels.api as sm
import pandas as pd
from numpy import log
# from datetime import datetime


lib_data = '/media/u70o/D/data'
lib_cpi_usa = os.path.join(lib_data, 'cpi_usa')

fname = 'CPIAUCSL.csv'
fpath_data = os.path.join(lib_cpi_usa, fname)

data = pd.read_csv(fpath_data, skiprows=10)  # FAME dataset

data.columns = ['date', 'CPIAUCSL']
data.loc[:, 'Pi'] = data.loc[:, 'CPIAUCSL'].div(data.loc[:, 'CPIAUCSL'].shift(1)).apply(log)*100

data.date = pd.to_datetime(data.date)
data.date = data.date + pd.DateOffset(1, months=1)  # this offset is needed
# checked the CPI inflation ended on 31 May 2024 against BLS new announcement.

# data = data.set_index('date')
data = data.set_index('date').asfreq('ME', method='bfill')

p = 0
d = 1
q = 1

fpath_arima = '/home/u70o/Documents/MATLAB/NRC/arima_%d%d%d_usa.csv' % (p, d, q)

if p == 1 and d == 0 and q == 1:
    trend = 'c'
if p == 0 and d == 1 and q == 1:
    trend = 't'


arima_model = sm.tsa.arima.ARIMA(
    data.Pi,
    exog=None,
    order=(p, d, q),
    seasonal_order=(0, 0, 0, 0),
    trend=trend,
    enforce_stationarity=True,
    enforce_invertibility=True,
    concentrate_scale=False,
    trend_offset=1,
    dates=data.index,
    freq='ME',
    missing='none',
    validate_specification=True)

res = arima_model.fit()

print(res.summary())
eps_name = 'epsilon_%d%d%d' % (p, d, q)
data.loc[:, eps_name] = res.forecasts_error.transpose()
data.loc[:, 'Pi_hat'] = data.loc[:, 'Pi'] - data.loc[:, 'epsilon_%d%d%d' % (p, d, q)]

data = data[['Pi', 'Pi_hat', eps_name]]
data = data.loc[data.notna().all(axis=1)]

data.to_csv(fpath_arima, index=True)

