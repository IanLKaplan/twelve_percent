
from datetime import datetime, timedelta

import matplotlib
from numpy import sqrt
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
from dateutil.relativedelta import relativedelta
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

data_source = 'yahoo'
# The start date is the date used in the examples in The 12% Solution
# yyyy-mm-dd
start_date_str = '2008-03-03'
start_date: datetime = datetime.fromisoformat(start_date_str)
look_back_date_str = '2007-12-03'
look_back_date: datetime = datetime.fromisoformat(look_back_date_str)
end_date: datetime = datetime.today() - timedelta(days=1)

etf_adjclose_file = 'equity_etf_adjclose'
equity_adj_close = get_market_data(file_name=etf_adjclose_file,
                                data_col='Adj Close',
                                symbols=equity_etfs,
                                data_source=data_source,
                                start_date=look_back_date,
                                end_date=end_date)

shy_adjclose_file = 'shy_adjclose'
shy_adj_close = get_market_data(file_name=shy_adjclose_file,
                                data_col='Adj Close',
                                symbols=[cash_etf],
                                data_source=data_source,
                                start_date=look_back_date,
                                end_date=end_date)

fixed_income_adjclose_file = "fixed_income_adjclose"
fixed_income_adjclose = get_market_data(file_name=fixed_income_adjclose_file,
                                data_col='Adj Close',
                                symbols=bond_etfs,
                                data_source=data_source,
                                start_date=look_back_date,
                                end_date=end_date)

# 13-week yearly treasury bond quote
risk_free_asset = '^IRX'
rf_file_name = 'rf_adj_close'
# The bond return is reported as a yearly return percentage
rf_adj_close = get_market_data(file_name=rf_file_name,
                                data_col='Adj Close',
                                symbols=[risk_free_asset],
                                data_source=data_source,
                                start_date=start_date,
                                end_date=end_date)

# The ^IRX interest rate is reported as a yearly percentage rate. Convert this to a daily interest rate
rf_adj_rate_np: np.array = np.array( rf_adj_close.values ) / 100
rf_daily_np = ((1 + rf_adj_rate_np) ** (1/360)) - 1
rf_daily_df: pd.DataFrame = pd.DataFrame( rf_daily_np, index=rf_adj_close.index, columns=['^IRX'])

corr_mat = round(equity_adj_close.corr(), 3)


print(tabulate(corr_mat, headers=[*corr_mat.columns], tablefmt='fancy_grid'))

asset_adj_close = equity_adj_close.copy()
asset_adj_close[shy_adj_close.columns[0]] = shy_adj_close


def findDateIndex(date_index: DatetimeIndex, search_date: datetime) -> int:
    '''
    In a DatetimeIndex, find the index of the date that is nearest to search_date.
    This date will either be equal to search_date or the next date that is less than
    search_date
    '''
    index: int = -1
    i = 0
    date_t = datetime.today()
    for i in range(0, len(date_index)):
        date_t = date_index[i]
        if type(date_t) == str:
            date_t = datetime.fromisoformat(date_t)
        if date_t >= search_date:
            break
    if date_t > search_date:
        index = i - 1
    elif date_t == search_date:
        index = i
    return index


start_date_ix = findDateIndex(asset_adj_close.index, start_date)

assert start_date_ix >= 0

def chooseAsset(start: int, end: int, asset_set: pd.DataFrame) -> pd.DataFrame:
    '''
    Choose an ETF asset or cash for a particular range of close price values.
    The ETF and cash time series should be contained in a single DataFrame
    The function returns a DataFrame with the highest returning asset for the
    period.
    '''
    rslt_df = asset_set
    if asset_set.shape[1] > 1:
        returns: pd.DataFrame = pd.DataFrame()
        for asset in asset_set.columns:
            t1 = asset_set[asset][start]
            t2 = asset_set[asset][end]
            r = (t2/t1) - 1
            returns[asset] = [r]
        column = returns.idxmax(axis=1)[0]
        rslt_df = pd.DataFrame(asset_set[column])
    return rslt_df

ts_df = chooseAsset(0, start_date_ix, asset_adj_close)

print(f'The asset for the first three month period will be {ts_df.columns[0]}')

last_quarter:pd.DataFrame = equity_adj_close[:][0:start_date_ix].copy()
last_quarter[shy_adj_close.columns[0]] = shy_adj_close

for col in last_quarter.columns:
    last_quarter[col] = last_quarter[col] - last_quarter[col][0]

last_quarter.plot(grid=True, title='4th Quarter 2007 Returns', figsize=(10,6))

def simple_return(time_series: np.array, period: int) -> List :
    return list(((time_series[i]/time_series[i-period]) - 1.0 for i in range(period, len(time_series), period)))


def return_df(time_series_df: pd.DataFrame) -> pd.DataFrame:
    r_df: pd.DataFrame = pd.DataFrame()
    time_series_a: np.array = time_series_df.values
    return_l = simple_return(time_series_a, 1)
    r_df = pd.DataFrame(return_l)
    date_index = time_series_df.index
    r_df.index = date_index[1:len(date_index)]
    r_df.columns = time_series_df.columns
    return r_df


def apply_return(start_val: float, return_df: pd.DataFrame) -> np.array:
    port_a: np.array = np.zeros( return_df.shape[0] + 1)
    port_a[0] = start_val
    return_a = return_df.values
    for i in range(1, len(port_a)):
        port_a[i] = port_a[i-1] + port_a[i-1] * return_a[i-1]
    return port_a


def portfolio_return(holdings: float,
                     asset_percent: float,
                     bond_percent: float,
                     asset_etfs: pd.DataFrame,
                     bond_etfs: pd.DataFrame,
                     start_date: datetime,
                     end_date: datetime,
                     year_rebalance: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    asset_holding= holdings * asset_percent
    bond_holding=holdings * bond_percent
    back_delta = relativedelta(months=3)
    forward_delta = relativedelta(months=1)
    date_index = asset_etfs.index
    start_date_i = start_date
    current_year = start_date.year
    investments = pd.DataFrame()
    portfolio_a = np.zeros(0)
    last_index = 0
    bond_asset_l = list()
    equity_asset_l = list()
    month_index_l = list()
    while start_date_i <= end_date:
        # Start of the back-test data
        back_start = start_date_i - back_delta
        # End of the back test data
        back_end = start_date_i
        # end of the forward data period (e.g., one month)
        forward_end = start_date_i + forward_delta
        start_ix = findDateIndex(date_index, back_start)
        end_ix = findDateIndex(date_index, back_end)
        forward_ix = findDateIndex(date_index, forward_end)
        if start_ix >= 0 and end_ix >= 0 and forward_ix >= 0:
            # Choose an asset based on the past three months
            asset_df = chooseAsset(start=start_ix, end=end_ix, asset_set=asset_etfs)
            asset_month_df = asset_df[:][end_ix:forward_ix]
            asset_return_df = return_df(asset_month_df)
            bond_df = chooseAsset(start=start_ix, end=end_ix, asset_set=bond_etfs)
            bond_asset = bond_df.columns[0]
            equity_asset = asset_df.columns[0]
            month_index = asset_month_df.index[0]
            bond_asset_l.append(bond_asset)
            equity_asset_l.append(equity_asset)
            month_index_l.append(month_index)
            bond_month_df = bond_df[:][end_ix:forward_ix]
            bond_return_df = return_df(bond_month_df)
            port_asset_a = apply_return(asset_holding, asset_return_df)
            port_bond_a = apply_return(bond_holding, bond_return_df)
            port_total_a = port_asset_a + port_bond_a
            portfolio_a = np.append(portfolio_a, port_total_a)
            last_index = forward_ix
            start_date_i = forward_end
            if year_rebalance:
                asset_holding = port_asset_a[-1]
                bond_holding = port_bond_a[-1]
                if start_date_i.year > current_year:
                    asset_holding = port_total_a[-1] * asset_percent
                    bond_holding = port_total_a[-1] * bond_percent
                    current_year = start_date_i.year
            else:
                asset_holding = port_total_a[-1] * asset_percent
                bond_holding = port_total_a[-1] * bond_percent
        else:
            break
    portfolio_df = pd.DataFrame(portfolio_a)
    portfolio_df.columns = ['portfolio']
    index_start = findDateIndex(date_index, start_date)
    date_index = asset_etfs.index
    portfolio_index = date_index[index_start:last_index]
    portfolio_df.index = portfolio_index
    choices_df = pd.DataFrame()
    choices_df['Equity'] = pd.DataFrame(equity_asset_l)
    choices_df['Bond'] = pd.DataFrame(bond_asset_l)
    choices_df.index = month_index_l
    return portfolio_df, choices_df


holdings = 100000
equity_percent = 0.6
bond_percent = 0.4

tlt = pd.DataFrame(fixed_income_adjclose['TLT'])
portfolio_df, assets_df = portfolio_return(holdings=holdings,
                                              asset_percent=equity_percent,
                                              bond_percent=bond_percent,
                                              asset_etfs=asset_adj_close,
                                              bond_etfs=tlt,
                                              start_date=start_date,
                                              end_date=end_date,
                                              year_rebalance=False)


def build_plot_data(holdings: float, portfolio_df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    port_start_date = portfolio_df.index[0]
    if type(port_start_date) == str:
        port_start_date = datetime.fromisoformat(port_start_date)
    port_end_date = portfolio_df.index[-1]
    if type(port_end_date) == str:
        port_end_date = datetime.fromisoformat(port_end_date)
    spy_index = spy_df.index
    spy_start_ix = findDateIndex(spy_index, port_start_date)
    spy_end_ix = findDateIndex(spy_index, port_end_date)
    spy_df = pd.DataFrame(spy_df[:][spy_start_ix:spy_end_ix+1])
    spy_return = return_df(spy_df)
    spy_return_a = apply_return(start_val=holdings, return_df=spy_return.copy())
    spy_port = pd.DataFrame(spy_return_a)
    spy_port.columns = ['SPY']
    spy_port.index = spy_df.index
    plot_df = portfolio_df.copy()
    plot_df['SPY'] = spy_port
    return plot_df


def adjust_time_series(ts_one_df: pd.DataFrame, ts_two_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adjust two DataFrame time series with overlapping date indices so that they
    are the same length with the same date indices.
    """
    ts_one_index = pd.to_datetime(ts_one_df.index)
    ts_two_index = pd.to_datetime(ts_two_df.index)
        # filter the close prices
    matching_dates = ts_one_index.isin( ts_two_index )
    ts_one_adj = ts_one_df[matching_dates]
    # filter the rf_prices
    ts_one_index = pd.to_datetime(ts_one_adj.index)
    matching_dates = ts_two_index.isin(ts_one_index)
    ts_two_adj = ts_two_df[matching_dates]
    return ts_one_adj, ts_two_adj


spy_df, portfolio_df = adjust_time_series(pd.DataFrame(equity_adj_close['SPY']), portfolio_df)
plot_df = build_plot_data(holdings, portfolio_df, spy_df)

trading_days = 252

spy_return = return_df(spy_df)
port_return = return_df(portfolio_df)
spy_volatility = round(spy_return.values.std() * sqrt(trading_days) * 100, 2)
port_volatility = round(port_return.values.std() * sqrt(trading_days) * 100, 2)

vol_df = pd.DataFrame([port_volatility, spy_volatility])
vol_df.columns = ['Standard Deviation']
vol_df.index = ['Portfolio', 'SPY']

print(tabulate(vol_df, headers=[*vol_df.columns], tablefmt='fancy_grid'))

def period_return(portfolio_df: pd.DataFrame, period: int) -> pd.DataFrame:
    date_index = portfolio_df.index
    values_a = portfolio_df.values
    date_list = list()
    return_list = list()
    for i in range(period, len(values_a), period):
        r = (values_a[i]/values_a[i-period]) - 1
        d = date_index[i]
        return_list.append(r)
        date_list.append(d)
    return_df = pd.DataFrame(return_list)
    return_df.index = date_list
    return return_df

period_return_df = period_return(portfolio_df=portfolio_df, period=trading_days)
period_return_df.columns = ['Yearly Return']

def excess_return_series(asset_return: pd.Series, risk_free: pd.Series) -> pd.DataFrame:
    excess_ret = asset_return.values.flatten() - risk_free.values.flatten()
    excess_ret_df = pd.DataFrame(excess_ret, index=asset_return.index)
    return excess_ret_df


def excess_return_df(asset_return: pd.DataFrame, risk_free: pd.Series) -> pd.DataFrame:
    excess_df: pd.DataFrame = pd.DataFrame()
    for col in asset_return.columns:
        e_df = excess_return_series(asset_return[col], risk_free)
        e_df.columns = [col]
        excess_df[col] = e_df
    return excess_df

def calc_sharpe_ratio(asset_return: pd.DataFrame, risk_free: pd.Series, period: int) -> pd.DataFrame:
    excess_return = excess_return_df(asset_return, risk_free)
    return_mean: List = []
    return_stddev: List = []
    for col in excess_return.columns:
        mu = np.mean(excess_return[col])
        std = np.std(excess_return[col])
        return_mean.append(mu)
        return_stddev.append(std)
    # daily Sharpe ratio
    # https://quant.stackexchange.com/questions/2260/how-to-annualize-sharpe-ratio
    sharpe_ratio = (np.asarray(return_mean) / np.asarray(return_stddev)) * np.sqrt(period)
    result_df: pd.DataFrame = pd.DataFrame(sharpe_ratio).transpose()
    result_df.columns = asset_return.columns
    ix = asset_return.index
    ix_start = ix[0]
    ix_end = ix[-1]
    if type(ix_start) == str:
        dateformat = '%Y-%m-%d'
        ix_start = datetime.strptime(ix_start, dateformat).date()
        ix_end = datetime.strptime(ix_end, dateformat).date()
    index_str = f'{ix_start} : {ix_end}'
    result_df.index = [ index_str ]
    return result_df


# Interest rates are quoted for the days when banks are open. The number of bank open days is less than
# the number of trading days. Adjust the portfolio_return series and the interest rate series so that they
# align.
rf_daily_adj, portfolio_return_adj = adjust_time_series(rf_daily_df, port_return)
spy_return_adj, t = adjust_time_series(spy_return, rf_daily_adj)

rf_daily_s = rf_daily_adj.squeeze()

portfolio_sharpe = calc_sharpe_ratio(portfolio_return_adj, rf_daily_s, trading_days)
spy_sharpe = calc_sharpe_ratio(spy_return_adj, rf_daily_s, trading_days)

sharpe_df = pd.concat([portfolio_sharpe, spy_sharpe], axis=1)

print(tabulate(sharpe_df, headers=[*sharpe_df.columns], tablefmt='fancy_grid'))

spy_adj_df, portfolio_adj_df = adjust_time_series(spy_df, portfolio_df)
spy_period_return_df = period_return(portfolio_df=spy_adj_df, period=trading_days)
portfolio_spy_return_df = pd.concat([period_return_df, spy_period_return_df], axis=1)
portfolio_spy_return_df.columns = ['ETF Rotation', 'SPY']
portfolio_spy_return_df = round(portfolio_spy_return_df * 100, 2)

print(tabulate(portfolio_spy_return_df, headers=[*portfolio_spy_return_df.columns], tablefmt='fancy_grid'))

average_return_df = pd.DataFrame(portfolio_spy_return_df.mean()).transpose()

print(tabulate(average_return_df, headers=[*average_return_df.columns], tablefmt='fancy_grid'))

new_equity_etfs = ['XLE', 'VUG', 'VBR', 'FXZ',
                   'VDC', 'VCR', 'VFH', 'VGT',
                   'VHT', 'SOXX']

new_bond_etfs = ['SPIP', 'BIV', 'IEF', 'VYM']

short_etfs = ['SDS', 'RWM']

new_etf_adjclose_file = 'new_equity_etf_adjclose'
new_equity_adj_close = get_market_data(file_name=new_etf_adjclose_file,
                                data_col='Adj Close',
                                symbols=new_equity_etfs,
                                data_source=data_source,
                                start_date=look_back_date,
                                end_date=end_date)

new_bond_adjclose_file = 'new_bond_etf_adjclose'
new_bond_adj_close = get_market_data(file_name=new_bond_adjclose_file,
                                data_col='Adj Close',
                                symbols=new_bond_etfs,
                                data_source=data_source,
                                start_date=look_back_date,
                                end_date=end_date)

short_etf_adjclose_file = 'short_etf_adjclose'
short_etf_adj_close = get_market_data(file_name=short_etf_adjclose_file,
                                data_col='Adj Close',
                                symbols=short_etfs,
                                data_source=data_source,
                                start_date=look_back_date,
                                end_date=end_date)

new_etf_set = pd.concat([new_equity_adj_close, equity_adj_close, shy_adj_close], axis=1)
new_bond_set = pd.concat([new_bond_adj_close, fixed_income_adjclose['TLT']], axis=1)

print("Hi there")