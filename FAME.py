
import os
import pandas as pd


def read_FAME(series_, lib_series_, generate_csvs_=False, dayfirst_=True):
    fname_ = series_ + '.xlsx'
    series_nm_, freq_ = series_.split('.')

    if not os.path.exists(os.path.join(lib_series_, series_nm_, freq_, 'metadata')):
        os.mkdir(os.path.join(lib_series_, series_nm_, freq_, 'metadata'))
    if not os.path.exists(os.path.join(lib_series_, series_nm_, freq_, 'with_metadata')):
        os.mkdir(os.path.join(lib_series_, series_nm_, freq_, 'with_metadata'))

    fpath_ = os.path.join(lib_series_, fname_)
    original_metadata_ = pd.read_excel(fpath_, nrows=8, header=None, index_col=0)
    metadata_ = original_metadata_.transpose()
    original_columns = metadata_.columns
    metadata_.columns = ['desc', 'units', 'sa', 'freq', 'action', 'src', 'pbase', 'series']
    metadata_ = metadata_.loc[:, list(metadata_.columns[-1:]) + list(metadata_.columns[:-1])]

    data_ = pd.read_excel(fpath_, skiprows=7, header=0)
    data_.columns = ['date'] + [col.upper() for col in data_.columns[1:]]
    data_.loc[:, 'date'] = pd.to_datetime(data_.loc[:, 'date'], dayfirst=dayfirst_)
    data_.set_index('date', inplace=True)
    idx_ = data_.isna().all(axis=1)
    data_ = data_.loc[~idx_]
    if generate_csvs_:
        for col in data_:
            idx_col_ = ~data_.loc[:, col].isna()
            fpath_csv_ = os.path.join(lib_series_, series_nm_, freq_, col + '.csv')
            fpath_metadata_csv_ = os.path.join(lib_series_, series_nm_, freq_, 'metadata', 'metadata_' + col + '.csv')
            fpath_with_metadata_csv_ = os.path.join(lib_series_, series_nm_, freq_, 'with_metadata', col + '.csv')

            data_.loc[idx_col_, col].to_csv(fpath_csv_, index=True)
            data_.loc[idx_col_, col].to_csv(fpath_with_metadata_csv_, index=True, header=False)

            original_metadata_.to_csv(fpath_metadata_csv_,
                                      index=True,
                                      header=False,
                                      columns=[list(data_.columns).index(col)+1])

            with open(fpath_metadata_csv_, 'r') as fp:
                txt = fp.read()
            with open(fpath_with_metadata_csv_, 'r') as fp:
                txt = txt + fp.read()
            with open(fpath_with_metadata_csv_, 'w') as fp:
                fp.write(txt)

        metadata_.to_csv(os.path.join(lib_series_, 'metadata_' + series_ + '.csv'), index=False)

    return data_, metadata_


# lib_data = '/media/u70o/D/data'
# freqs = ['M', 'D']
# seriess = ['TSB_BAGR_MAKAM', 'TSB_BAGR_GALIL', 'TSB_BAGR_GILON', 'TSB_BAGR_SHACHAR']
# # seriess = ['TSB_BAGR_MAKAM', ]
# # seriess = ['SM', ]
#
# for freq in freqs:
#     for series in seriess:
#         series = series + '.%s' % freq
#         lib_series = os.path.join(lib_data, 'Israel')
#         data, metadata = read_FAME(series, lib_series, generate_csvs_=True)
#

