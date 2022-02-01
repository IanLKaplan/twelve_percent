from datetime import datetime, timedelta

import matplotlib
from tabulate import tabulate
from typing import List, Tuple
from pandas_datareader import data
import pypfopt as pyopt
from pypfopt import expected_returns
from pypfopt import risk_models
from pypfopt import plotting, CLA
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from pandas.core.indexes.datetimes import DatetimeIndex
import numpy as np
from pathlib import Path
import tempfile
import quantstats as qs

def get_market_data(file_name: str,
                    data_col: str,
                    symbols: List,
                    data_source: str,
                    start_date: datetime,
                    end_date: datetime) -> pd.DataFrame:
    """
      file_name: the file name in the temp directory that will be used to store the data
      data_col: the type of data - 'Adj Close', 'Close', 'High', 'Low', 'Open', Volume'
      symbols: a list of symbols to fetch data for
      data_source: yahoo, etc...
      start_date: the start date for the time series
      end_date: the end data for the time series
      Returns: a Pandas DataFrame containing the data.

      If a file of market data does not already exist in the temporary directory, fetch it from the
      data_source.
    """
    temp_root: str = tempfile.gettempdir() + '/'
    file_path: str = temp_root + file_name
    temp_file_path = Path(file_path)
    file_size = 0
    if temp_file_path.exists():
        file_size = temp_file_path.stat().st_size

    if file_size > 0:
        close_data = pd.read_csv(file_path, index_col='Date')
    else:
        panel_data: pd.DataFrame = data.DataReader(symbols, data_source, start_date, end_date)
        close_data: pd.DataFrame = panel_data[data_col]
        close_data.to_csv(file_path)
    assert len(close_data) > 0, f'Error reading data for {symbols}'
    return close_data


plt.style.use('seaborn-whitegrid')

equity_etfs = ['IWM', 'MDY', 'QQQ', 'SPY']
bond_etfs = ['JNK', 'TLT']
cash_etf = 'SHY'

trading_days = 252
days_in_quarter = trading_days // 4
days_in_month = trading_days // 12

data_source = 'yahoo'
# The start date is the date used in the examples in The 12% Solution
# yyyy-mm-dd
start_date_str = '2008-01-01'
start_date: datetime = datetime.fromisoformat(start_date_str)
end_date: datetime = datetime.today() - timedelta(days=1)

equity_etf_file = 'equity_etf_close'

etf_close = get_market_data(file_name=equity_etf_file,
                                data_col='Close',
                                symbols=equity_etfs,
                                data_source=data_source,
                                start_date=start_date  - timedelta(days=(365//4)),
                                end_date=end_date)

shy_adjclose_file = 'shy_adjclose'
shy_adj_close = get_market_data(file_name=shy_adjclose_file,
                                data_col='Adj Close',
                                symbols=[cash_etf],
                                data_source=data_source,
                                start_date=start_date - timedelta(days=(365//4)),
                                end_date=end_date)

shy_close_file = 'shy_close'
shy_close = get_market_data(file_name=shy_close_file,
                                data_col='Close',
                                symbols=[cash_etf],
                                data_source=data_source,
                                start_date=start_date,
                                end_date=end_date)

fixed_income_adjclose_file = "fixed_income_adjclose"
fixed_income_adjclose = get_market_data(file_name=fixed_income_adjclose_file,
                                data_col='Adj Close',
                                symbols=bond_etfs,
                                data_source=data_source,
                                start_date=start_date,
                                end_date=end_date)

fixed_income_close_file = "fixed_income_close"
fixed_income_close = get_market_data(file_name=fixed_income_close_file,
                                data_col='Close',
                                symbols=bond_etfs,
                                data_source=data_source,
                                start_date=start_date,
                                end_date=end_date)

corr_mat = round(etf_close.corr(), 3)
print(tabulate(corr_mat, headers=[*corr_mat.columns], tablefmt='fancy_grid'))


def findDateIndex(ix: DatetimeIndex, search_date: datetime) -> int:
    index: int = -1
    for i, date in enumerate(ix):
        date_t = datetime.fromisoformat(date)
        if date_t == search_date:
            index = i
            break
    return index

newyears_2008_ix = findDateIndex(etf_close.index, datetime.fromisoformat('2008-01-03'))

assert newyears_2008_ix >= 0

def chooseAsset(start: int, end: int, etf_set: pd.DataFrame, cash: pd.DataFrame) -> pd.DataFrame:
    returns: pd.DataFrame = pd.DataFrame()
    for asset in etf_set.columns:
        t1 = etf_set[asset][start]
        t2 = etf_set[asset][end]
        r = (t2/t1) - 1
        returns[asset] = [r]
    cash_t1 = cash[cash.columns[0]][start]
    cash_t2 = cash[cash.columns[0]][end]
    cash_ret = (t2/t1) - 1
    max_ret = returns.max(axis=1)
    rslt_df = cash
    if float(max_ret) > cash_ret:
        for asset in returns.columns:
            if returns[asset][0] == float(max_ret):
                rslt_df = pd.DataFrame(etf_set[asset])
    return rslt_df

ts_df = chooseAsset(0, newyears_2008_ix, etf_close, shy_adj_close)


print("Hi there")