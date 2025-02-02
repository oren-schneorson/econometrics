
import os
import pandas as pd
from datetime import datetime
from numpy import log, array

import matlab.engine
eng = matlab.engine.start_matlab()

# user = 'u70o'
user = 'oren-laptop'

lib_data = '/media/%s/D/data' % user
lib_israel = os.path.join(lib_data, 'Israel')


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
