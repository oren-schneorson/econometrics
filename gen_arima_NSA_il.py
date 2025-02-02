
import os.path
import statsmodels.api as sm
import pandas as pd
from numpy import log, sqrt
# from datetime import datetime
# from FAME import read_FAME
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

matplotlib.use('QtAgg')
# ['GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX', 'nbAgg', 'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg',
# 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']

lib_data = '/media/u70o/D/data'
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

fname = 'CP_NSA.M.csv'
fpath_data = os.path.join(lib_israel, fname)

data = pd.read_csv(fpath_data, skiprows=7)  # FAME dataset
data = data.loc[data[data.columns[-1]].notna()]

data.columns = ['date', 'CP_SA']

# data.loc[:, 'Pi'] = data.loc[:, 'CP_SA'].div(data.loc[:, 'CP_SA'].shift(1)).apply(log)*100
data.loc[:, 'Pi'] = data.loc[:, 'CP_SA'].div(data.loc[:, 'CP_SA'].shift(1)) - 1
data.loc[:, 'Pi'] = data.loc[:, 'Pi'] * 100
data = pd.merge(data, INF_TRGT)
data.loc[:, 'Pi'] = data.loc[:, 'Pi'] - data.loc[:, 'INF_TRGT']  # deviation from target (detrend)

data.date = pd.to_datetime(data.date, dayfirst=True)
data = data.set_index('date').asfreq('ME', method='bfill')

data = data.loc[data.index > datetime(1992, 1, 1)]

# data['Pi'].plot()
# data['Pi'].mean()
# plt.show()

p = 0
d = 1
q = 1

sp = 0
sd = 1
sq = 1

seasons = 12

df = []
cols_df = ['sarima_typ', 'p', 'd', 'q', 'sp', 'sd', 'sq', 'seasons', 'aic', 'rmse']

for p in range(2, 4):
    for d in range(1):
        for q in range(1, 4):
            for sp in range(1):
                for sd in range(1, 2):
                    for sq in range(1, 3):

                        sarima_params = (p, d, q, sp, sd, sq, seasons)
                        sarima_typ = 'sarima_NSA_(%d,%d,%d)(%d,%d,%d,%d)_il' % sarima_params
                        fpath_arima = '/home/u70o/Documents/MATLAB/NRC/%s.csv' % sarima_typ

                        if d == 0:
                            trend = 'c'
                        else:
                            trend = 't'

                        arima_model = sm.tsa.arima.ARIMA(
                            data.Pi,
                            exog=None,
                            order=(p, d, q),
                            seasonal_order=(sp, sd, sq, seasons),
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

                        data.loc[:, 'forecasts_error'] = res.forecasts_error.transpose()

                        idx = data.index >= datetime(1996, 2, 29)
                        rmse = sqrt(data.loc[idx, 'forecasts_error'].iloc[1:].apply(lambda x: x ** 2).mean())

                        row = (sarima_typ, ) + sarima_params + (res.aic, rmse)
                        df.append(row)

df = pd.DataFrame(df, columns=cols_df)
df = df.sort_values(by='rmse', ascending=True)
print(df)

raise ValueError
print(res.summary())
eps_name = 'epsilon_%d%d%d' % (p, d, q)
data.loc[:, eps_name] = res.forecasts_error.transpose()

data.loc[:, 'Pi'] = data.loc[:, 'Pi'] + data.loc[:, 'INF_TRGT']
data.loc[:, 'Pi_hat'] = data.loc[:, 'Pi'] - data.loc[:, eps_name]
data = data.drop(['INF_TRGT'], axis=1)


data.loc[:, 'Pi_hat'] = data.loc[:, 'Pi'] - data.loc[:, 'epsilon_%d%d%d' % (p, d, q)]

data = data[['Pi', 'Pi_hat', eps_name]]
data = data.loc[data.notna().all(axis=1)]
# data.to_csv(fpath_arima, index=True)

data.loc[:, 'ma'] = data.loc[:, eps_name].rolling(60).mean()


for lag in list(range(1, 37)):
    print(lag, data.loc[:, eps_name].iloc[1:].autocorr(lag))

data.loc[:, 'ma'].iloc[1:].autocorr(1)


data.loc[:, eps_name].iloc[1:].mean()
print(sqrt(data.loc[:, eps_name].iloc[1:].apply(lambda x: x ** 2).mean()))  # .34 deviation from trgt


# data.loc[:, 'ma'].plot()
# data[[eps_name, 'ma']].plot()
# plt.show(block=True)

# UITB = pd.read_excel('/media/u70o/D/data/Israel/UITB.xlsx')
# pd.plotting.autocorrelation_plot(data.loc[:, eps_name].iloc[1:]).set_xlim([0, 24])
# pd.plotting.autocorrelation_plot(UITB.UITB).set_xlim([0, 24])



'''
Subtracting INF_TRGT from Pi, and then attempting to predict Pi it results in a UIAR with 0 autocorrelation.
This should be very close to the TB model, where the TBill is doing the job of INF_TRGT to some extent.
'''
