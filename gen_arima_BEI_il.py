
import os.path
import statsmodels.api as sm
import pandas as pd
from numpy import log, sqrt, diff
# from datetime import datetime
from FAME import read_FAME
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('QtAgg')
# ['GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX', 'nbAgg', 'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg',
# 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']

user = input('Please insert username.')

lib_data = '/media/%s/D/data' % user
lib_israel = os.path.join(lib_data, 'Israel')

fname = 'INF_TRGT.D.xlsx'
fpath_data = os.path.join(lib_israel, fname)

INF_TRGT = pd.read_excel(fpath_data, skiprows=7)  # FAME dataset
INF_TRGT.columns = ['date', ] + list(INF_TRGT.columns[1:])

cols_trgt = ['INF_MIN_TRGT.D', 'INF_MAX_TRGT.D']
INF_TRGT.loc[:, 'INF_TRGT'] = INF_TRGT[cols_trgt].mean(axis=1) / 12
INF_TRGT = INF_TRGT.drop(cols_trgt, axis=1)
INF_TRGT.columns = ['date', 'INF_TRGT']

INF_TRGT.date = pd.to_datetime(INF_TRGT.date, dayfirst=False)
INF_TRGT = INF_TRGT.set_index('date').asfreq('ME', method='bfill')
INF_TRGT.loc[:, 'date'] = INF_TRGT.index.strftime('%d/%m/%Y')
INF_TRGT = INF_TRGT.reset_index(drop=True)


TELBOR, _ = read_FAME('TELBOR.M', lib_israel, False)
TELBOR = TELBOR.asfreq('ME')
TELBOR = TELBOR.shift(periods=1, freq='ME')

TSB_ZRD, _ = read_FAME('TSB_ZRD.D', lib_israel, False)
TSB_ZRD = TSB_ZRD.asfreq('d').ffill().asfreq('ME')
TSB_ZRD = TSB_ZRD.shift(periods=1, freq='ME')
BEI = pd.merge(TELBOR[['BL.TELBOR_01M.M']], TSB_ZRD[['TSB_ZRD_01Y.D']], left_index=True, right_index=True)
BEI = BEI/12
BEI.loc[:, 'BEI'] = BEI.loc[:, 'BL.TELBOR_01M.M'] - BEI.loc[:, 'TSB_ZRD_01Y.D']


fname = 'CP_SA.M.csv'
fpath_data = os.path.join(lib_israel, fname)

data = pd.read_csv(fpath_data, skiprows=7)  # FAME dataset
data = data.loc[data[data.columns[-1]].notna()]

data.columns = ['date', 'CP_SA']

# data.loc[:, 'Pi'] = data.loc[:, 'CP_SA'].div(data.loc[:, 'CP_SA'].shift(1)).apply(log)*100
data.loc[:, 'Pi'] = data.loc[:, 'CP_SA'].div(data.loc[:, 'CP_SA'].shift(1)) - 1
data.loc[:, 'Pi'] = data.loc[:, 'Pi'] * 100
# data = pd.merge(data, INF_TRGT)
# data.loc[:, 'Pi'] = data.loc[:, 'Pi'] - data.loc[:, 'INF_TRGT']  # deviation from target (detrend)

data.date = pd.to_datetime(data.date, dayfirst=True)
data = data.set_index('date').asfreq('ME', method='bfill')

data = pd.merge(data, BEI[['BEI']], left_index=True, right_index=True)
data.loc[:, 'err'] = data.Pi - data.BEI


# data['Pi'].plot()
# data['Pi'].mean()
# plt.show()

p = 0
d = 0
q = 1

fpath_arima = '/home/%s/Documents/MATLAB/NRC/arima_BEI_%d%d%d_il.csv' % (user, p, d, q)

if d == 0:
    trend = 'c'
else:
    trend = 't'


arima_model = sm.tsa.arima.ARIMA(
    data.err,
    exog=None,
    order=(p, d, q),
    seasonal_order=(0, 0, 0, 0),
    trend=None,
    enforce_stationarity=True,
    enforce_invertibility=True,
    concentrate_scale=False,
    trend_offset=1,
    dates=data.index,
    freq='ME',
    missing='none',
    validate_specification=True)

res = arima_model.fit()

data.loc[:, 'err_fe'] = res.forecasts_error[0]
data.loc[:, 'a'] = data.err.diff()-data.loc[:, 'err_fe'].diff()
data.loc[:, 'alpha'] = data.a.cumsum()
data.loc[:, 'alpha'].plot()
data.loc[:, 'a'].plot()
plt.show(block=False)


data.alpha.shift(-1)

data.loc[:, 'b'] = data.Pi - data.BEI-data.alpha
data.loc[:, 'b'] = data.Pi - data.BEI-data.alpha.shift(-1)
data.loc[:, 'b'].mean()
sqrt(data.loc[:, 'b'].apply(lambda x: x ** 2).mean())
data.b.rolling(60).mean().plot()
plt.show(block=False)


data.loc[:, 'fe'] = res.forecasts_error[0]  # out of sample forecast error (one step ahead)

sqrt(data.loc[:, 'fe'].iloc[1:].apply(lambda x: x ** 2).mean())  # .3 deviation from trgt

data.Pi - (data.BEI + data.err)
data.loc [:, 'EITB_{t-1}'] = (data.BEI + res.forecasts[0])
data.loc [:, 'UITB_{t}'] = data.Pi - data.loc [:, 'EITB_{t-1}']
data.loc [:, 'UITB_{t}'].mean()
sqrt(data.loc [:, 'UITB_{t}'].apply(lambda x: x**2).mean())



# data = data[['Pi', 'Pi_hat', eps_name]]
# data = data.loc[data.notna().all(axis=1)]
# data.to_csv(fpath_arima, index=True)

data.loc[:, 'ma'] = data.loc[:, 'UITB_{t}'].rolling(60).mean()


for lag in list(range(1, 37)):
    print(lag, data.loc[:, 'UITB_{t}'].iloc[1:].autocorr(lag))

data.loc[:, 'ma'].iloc[1:].autocorr(1)


data.loc[:, 'UITB_{t}'].iloc[1:].mean()
sqrt(data.loc[:, 'UITB_{t}'].iloc[1:].apply(lambda x: x ** 2).mean())  # .34 deviation from trgt


data.loc[:, 'ma'].plot()
# data[[eps_name, 'ma']].plot()
plt.show(block=True)

# UITB = pd.read_excel('/media/%s/D/data/Israel/UITB.xlsx' % user)
# pd.plotting.autocorrelation_plot(data.loc[:, eps_name].iloc[1:]).set_xlim([0, 24])
# pd.plotting.autocorrelation_plot(UITB.UITB).set_xlim([0, 24])



'''
Subtracting INF_TRGT from Pi, and then attempting to predict Pi it results in a UIAR with 0 autocorrelation.
This should be very close to the TB model, where the TBill is doing the job of INF_TRGT to some extent.
'''
