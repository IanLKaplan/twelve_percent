
from datetime import datetime, timedelta
from numpy import sqrt
from tabulate import tabulate
from typing import List, Tuple
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.indexes.datetimes import DatetimeIndex
from dateutil.relativedelta import relativedelta
import numpy as np
from pathlib import Path
import tempfile

plt.style.use('seaborn-whitegrid')
pd.options.mode.chained_assignment = 'raise'


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
# get rid of any time component
end_date = datetime(end_date.year, end_date.month, end_date.day)

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

# The ^IRX interest rate is reported as a yearly percentage rate.
# Convert this to a daily interest rate
rf_adj_rate_np: np.array = np.array( rf_adj_close.values ) / 100
rf_daily_np = ((1 + rf_adj_rate_np) ** (1/360)) - 1
rf_daily_df: pd.DataFrame = pd.DataFrame( rf_daily_np, index=rf_adj_close.index, columns=['^IRX'])

corr_mat = round(equity_adj_close.corr(), 3)

print(tabulate(corr_mat, headers=[*corr_mat.columns], tablefmt='fancy_grid'))


def convert_date(some_date):
    if type(some_date) == str:
        some_date = datetime.fromisoformat(some_date)
    elif type(some_date) == np.datetime64:
        ts = (some_date - np.datetime64('1970-01-01T00:00')) / np.timedelta64(1, 's')
        some_date = datetime.utcfromtimestamp(ts)
    return some_date


def findDateIndex(date_index: DatetimeIndex, search_date: datetime) -> int:
    '''
    In a DatetimeIndex, find the index of the date that is nearest to search_date.
    This date will either be equal to search_date or the next date that is less than
    search_date
    '''
    index: int = -1
    i = 0
    search_date = convert_date(search_date)
    date_t = datetime.today()
    for i in range(0, len(date_index)):
        date_t = convert_date(date_index[i])
        if date_t >= search_date:
            break
    if date_t > search_date:
        index = i - 1
    else:
        index = i
    return index

# A unit test for findDateIndex, which is a function that is used throughout the code.
test_date_index = DatetimeIndex(['2007-12-03', '2007-12-04', '2007-12-05', '2007-12-06', '2007-12-07',
                   '2007-12-10', '2007-12-11', '2007-12-12', '2007-12-13', '2007-12-14',
                   '2007-12-15', '2007-12-18', '2007-12-19'])

assert findDateIndex(test_date_index, datetime.fromisoformat('2007-12-03')) == 0
assert findDateIndex(test_date_index, datetime.fromisoformat('2007-12-09')) == 4
assert findDateIndex(test_date_index, datetime.fromisoformat('2007-12-14')) == 9
assert findDateIndex(test_date_index, datetime.fromisoformat('2007-12-17')) == 10
assert findDateIndex(test_date_index, datetime.fromisoformat('2007-12-20')) == 12


asset_adj_close = equity_adj_close.copy()
asset_adj_close[shy_adj_close.columns[0]] = shy_adj_close

start_date_ix = findDateIndex(asset_adj_close.index, start_date)

assert start_date_ix >= 0


def chooseAsset(start: int, end: int, asset_set: pd.DataFrame) -> pd.DataFrame:
    '''
    Choose an ETF asset for a particular range of close price values.
    The ETF time series should be contained in a single DataFrame
    The function returns a DataFrame with the highest returning asset for the
    period.
    '''
    rslt_df = asset_set
    asset_name = asset_set.columns[0]
    if asset_set.shape[1] > 1:
        ret_list = []
        start_date = asset_set.index[start]
        end_date = asset_set.index[end]
        for asset in asset_set.columns:
            ts = asset_set[asset][start:end+1]
            start_val = ts[0]
            end_val = ts[-1]
            r = (end_val/start_val) - 1
            ret_list.append(r)
        ret_df = pd.DataFrame(ret_list).transpose()
        ret_df.columns = asset_set.columns
        column = ret_df.idxmax(axis=1)[0]
        asset_name = column
        rslt_df = pd.DataFrame(asset_set[column])
    return rslt_df


def chooseAssetName(start: int, end: int, asset_set: pd.DataFrame) -> str:
    '''
    Choose an ETF asset or cash for a particular range of close price values.
    The ETF and cash time series should be contained in a single DataFrame
    The function returns a DataFrame with the highest returning asset for the
    period.
    '''
    asset_columns = asset_set.columns
    asset_name = asset_columns[0]
    if len(asset_columns) > 1:
        ret_list = []
        start_date = asset_set.index[start]
        end_date = asset_set.index[end]
        for asset in asset_set.columns:
            ts = asset_set[asset][start:end+1]
            start_val = ts[0]
            end_val = ts[-1]
            r = (end_val/start_val) - 1
            ret_list.append(r)
        ret_df = pd.DataFrame(ret_list).transpose()
        ret_df.columns = asset_set.columns
        column = ret_df.idxmax(axis=1)[0]
        asset_name = column
    return asset_name


start_date_ix = findDateIndex(asset_adj_close.index, start_date)
asset_name = chooseAssetName(0, start_date_ix, asset_adj_close)

print(f'The asset for the first three month period will be {asset_name}')



def percent_return_df(start_date: datetime, end_date: datetime, prices_df: pd.DataFrame) -> pd.DataFrame:
    def percent_return(time_series: pd.Series) -> pd.Series:
        return list(((time_series[i] / time_series[0]) - 1.0 for i in range(0, len(time_series))))


    date_index = prices_df.index
    start_ix = findDateIndex(date_index, start_date)
    end_ix = findDateIndex(date_index, end_date)
    period_df = prices_df[:][start_ix:end_ix+1]
    period_return_df = pd.DataFrame()
    for col in period_df.columns:
        return_series = percent_return(period_df[col])
        period_return_df[col] = return_series
    period_return_df.index = period_df.index
    return_percent_df = round(period_return_df * 100, 2)
    return return_percent_df

quarter_return_df = percent_return_df(start_date=look_back_date, end_date=start_date, prices_df=asset_adj_close)


def simple_return(time_series: np.array, period: int = 1) -> List :
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


def find_month_periods(start_date: datetime, end_date:datetime, data: pd.DataFrame) -> pd.DataFrame:
    start_date = convert_date(start_date)
    end_date = convert_date(end_date)
    date_index = data.index
    start_ix = findDateIndex(date_index, start_date)
    end_ix = findDateIndex(date_index, end_date)
    start_l = list()
    end_l = list()
    cur_month = start_date.month
    start_l.append(start_ix)
    i = 0
    for i in range(start_ix, end_ix+1):
        date_i = convert_date(date_index[i])
        if date_i.month != cur_month:
            end_l.append(i-1)
            start_l.append(i)
            cur_month = date_i.month
    end_l.append(i)
    # if there is note something like a full month period, remove the last period
    if end_l[-1] - start_l[-1] < 18:
        end_l.pop()
        start_l.pop()
    start_df = pd.DataFrame(start_l)
    end_df = pd.DataFrame(end_l)
    start_date_df = pd.DataFrame(date_index[start_l])
    end_date_df = pd.DataFrame(date_index[end_l])
    periods_df = pd.concat([start_df, start_date_df, end_df, end_date_df], axis=1)
    periods_df.columns = ['start_ix', 'start_date', 'end_ix', 'end_date']
    return periods_df


def portfolio_return(holdings: float,
                     asset_percent: float,
                     bond_percent: float,
                     asset_etfs: pd.DataFrame,
                     bond_etfs: pd.DataFrame,
                     start_date: datetime,
                     end_date: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    An implementation of the 12% Solution ETF rotation. From start_date to end_date
    the code calculates the highest past three month return from the asset_etfs and bond_etfs. The
    ETF with the highest return is selected for the next month.

    Calculate the monthly period from start_date to end_date. This will provide the monthly periods
    for the calculation.

    The daily returns are calculated for the next month and applied to the current portfolio balance.
    Then the loop steps forward by one month.
    """
    assert asset_etfs.shape[0] == bond_etfs.shape[0]
    periods_df = find_month_periods(start_date, end_date, asset_etfs)
    back_delta = relativedelta(months=3)
    date_index = asset_etfs.index
    bond_asset_l = list()
    equity_asset_l = list()
    month_index_l = list()
    portfolio_a = np.zeros(0)
    for row in range(periods_df.shape[0]):
        asset_holdings = holdings * asset_percent
        bond_holdings = holdings * bond_percent
        period_info = periods_df[:][row:row+1]
        month_start_date = convert_date(period_info['start_date'].values[0])
        back_start_date = month_start_date - back_delta
        back_start_ix = findDateIndex(date_index, back_start_date)
        month_start_ix = period_info['start_ix'].values[0]
        month_end_ix = period_info['end_ix'].values[0]
        equity_asset = chooseAssetName(start=back_start_ix, end=month_start_ix, asset_set=asset_etfs)
        bond_asset = chooseAssetName(start=back_start_ix, end=month_start_ix, asset_set=bond_etfs)
        equity_asset_l.append(equity_asset)
        bond_asset_l.append(bond_asset)
        month_index_l.append(month_start_date)
        asset_month_prices_df = pd.DataFrame(asset_etfs[equity_asset][month_start_ix:month_end_ix + 1])
        bond_month_prices_df = pd.DataFrame(bond_etfs[bond_asset][month_start_ix:month_end_ix + 1])
        asset_month_return_df = return_df(asset_month_prices_df)
        bond_month_return_df = return_df(bond_month_prices_df)
        asset_month_a = apply_return(asset_holdings, asset_month_return_df)
        bond_month_a = apply_return(bond_holdings, bond_month_return_df)
        portfolio_total_a = asset_month_a + bond_month_a
        holdings = portfolio_total_a[-1]
        portfolio_a = np.append(portfolio_a, portfolio_total_a)
    portfolio_df = pd.DataFrame(portfolio_a)
    portfolio_df.columns = ['portfolio']
    num_rows = periods_df.shape[0]
    first_row = periods_df[:][0:1]
    last_row = periods_df[:][num_rows - 1:num_rows]
    start_ix = first_row['start_ix'].values[0]
    end_ix = last_row['end_ix'].values[0]
    portfolio_index = date_index[start_ix:end_ix + 1]
    portfolio_df.index = portfolio_index
    choices_df = pd.DataFrame()
    choices_df['Equity'] = pd.DataFrame(equity_asset_l)
    choices_df['Bond'] = pd.DataFrame(bond_asset_l)
    choices_df.index = month_index_l
    return portfolio_df, choices_df


def portfolio_replay(holdings: float,
                     asset_percent: float,
                     bond_percent: float,
                     asset_etfs: pd.DataFrame,
                     bond_etfs: pd.DataFrame,
                     etf_selection: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the portfolio time series from a pre-defined set of ETFs (e.g., equity and bond)

    The etf_selection_df DataFrame has the following columns:
    start_ix  start_date  end_ix    end_date equity bond
    """
    assert asset_etfs.shape[0] == bond_etfs.shape[0]
    date_index = asset_etfs.index
    portfolio_a = np.zeros(0)
    for index, month_allocation in etf_selection.iterrows():
        asset_holdings = holdings * asset_percent
        bond_holdings = holdings * bond_percent
        month_start_ix = month_allocation['start_ix']
        month_end_ix = month_allocation['end_ix']
        equity_asset = month_allocation['equity']
        bond_asset = month_allocation['bond']
        asset_month_prices_df = pd.DataFrame(asset_etfs[equity_asset][month_start_ix:month_end_ix + 1])
        bond_month_prices_df = pd.DataFrame(bond_etfs[bond_asset][month_start_ix:month_end_ix + 1])
        asset_month_return_df = return_df(asset_month_prices_df)
        bond_month_return_df = return_df(bond_month_prices_df)
        asset_month_a = apply_return(asset_holdings, asset_month_return_df)
        bond_month_a = apply_return(bond_holdings, bond_month_return_df)
        portfolio_total_a = asset_month_a + bond_month_a
        holdings = portfolio_total_a[-1]
        portfolio_a = np.append(portfolio_a, portfolio_total_a)
    portfolio_df = pd.DataFrame(portfolio_a)
    portfolio_df.columns = ['portfolio']
    num_rows = etf_selection.shape[0]
    first_row = etf_selection[:][0:1]
    last_row = etf_selection[:][num_rows - 1:num_rows]
    start_ix = first_row['start_ix'].values[0]
    end_ix = last_row['end_ix'].values[0]
    portfolio_index = date_index[start_ix:end_ix+1]
    portfolio_df.index = portfolio_index
    return portfolio_df


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
                                              end_date=end_date)


def build_plot_data(holdings: float, portfolio_df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    port_start_date = portfolio_df.index[0]
    port_start_date = convert_date(port_start_date)
    port_end_date = portfolio_df.index[-1]
    port_end_date = convert_date(port_end_date)
    spy_index = spy_df.index
    spy_start_ix = findDateIndex(spy_index, port_start_date)
    spy_end_ix = findDateIndex(spy_index, port_end_date)
    spy_df = pd.DataFrame(spy_df[:][spy_start_ix:spy_end_ix+1])
    spy_return = return_df(spy_df)
    spy_return_a = apply_return(start_val=holdings, return_df=spy_return)
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


d2019_start = datetime.fromisoformat("2019-01-02")
d2019_end = datetime.fromisoformat("2019-12-31")

#
# Test for build_plot_data to make sure that the SPY component is handled correctly.
#
spy_unadj = pd.DataFrame(asset_adj_close['SPY'])
start_ix = findDateIndex(spy_unadj.index, d2019_start)
end_ix = findDateIndex(spy_unadj.index, d2019_end)
start_val = spy_unadj[spy_unadj.columns[0]].values[start_ix]
spy_section = pd.DataFrame(spy_unadj[spy_unadj.columns[0]][start_ix:end_ix+1])
spy_section.columns = ['portfolio']
plot_df = build_plot_data(start_val, spy_section, spy_unadj)
portfolio_a = plot_df[plot_df.columns[0]].values
spy_a = plot_df[plot_df.columns[1]].values
assert all( np.round(portfolio_a, 2) == np.round(spy_a, 2) )
#
# End test
#

spy_df = pd.DataFrame(equity_adj_close['SPY'])
spy_df, portfolio_df = adjust_time_series(spy_df, portfolio_df)
plot_df = build_plot_data(holdings, portfolio_df, spy_df)


trading_days = 252

spy_return = return_df(spy_df)
port_return = return_df(portfolio_df)
spy_volatility = round(spy_return.values.std() * sqrt(trading_days) * 100, 2)
port_volatility = round(port_return.values.std() * sqrt(trading_days) * 100, 2)

vol_df = pd.DataFrame([port_volatility, spy_volatility])
vol_df.columns = ['Yearly Standard Deviation (percent)']
vol_df.index = ['Portfolio', 'SPY']


print(tabulate(vol_df, headers=[*vol_df.columns], tablefmt='fancy_grid'))


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
    dateformat = '%Y-%m-%d'
    ix_start = ix[0]
    ix_start = convert_date(ix_start)
    ix_end = ix[len(ix)-1]
    ix_end = convert_date(ix_end)
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
spy_period_return_df = period_return(portfolio_df=spy_df, period=trading_days)
portfolio_spy_return_df = pd.concat([period_return_df, spy_period_return_df], axis=1)
portfolio_spy_return_df.columns = ['ETF Rotation', 'SPY']
portfolio_spy_return_df = round(portfolio_spy_return_df * 100, 2)

print(tabulate(portfolio_spy_return_df, headers=[*portfolio_spy_return_df.columns], tablefmt='fancy_grid'))

average_return_df = pd.DataFrame(portfolio_spy_return_df.mean()).transpose()

print(tabulate(average_return_df, headers=[*average_return_df.columns], tablefmt='fancy_grid'))

portfolio_df, assets_df = portfolio_return(holdings=holdings,
                                              asset_percent=equity_percent,
                                              bond_percent=bond_percent,
                                              asset_etfs=asset_adj_close,
                                              bond_etfs=fixed_income_adjclose,
                                              start_date=start_date,
                                              end_date=end_date)

plot_df = build_plot_data(holdings, portfolio_df, spy_df)

port_return = return_df(portfolio_df)
port_volatility = round(port_return.values.std() * sqrt(trading_days) * 100, 2)

vol_df = pd.DataFrame([port_volatility, spy_volatility])
vol_df.columns = ['Yearly Standard Deviation (percent)']
vol_df.index = ['Portfolio', 'SPY']

print(tabulate(vol_df, headers=[*vol_df.columns], tablefmt='fancy_grid'))

rf_daily_adj, portfolio_return_adj = adjust_time_series(rf_daily_df, port_return)
portfolio_sharpe = calc_sharpe_ratio(portfolio_return_adj, rf_daily_s, trading_days)
sharpe_df = pd.concat([portfolio_sharpe, spy_sharpe], axis=1)
print("Sharpe Ratio:")
print(tabulate(sharpe_df, headers=[*sharpe_df.columns], tablefmt='fancy_grid'))

period_return_bond_df = period_return(portfolio_df=portfolio_df, period=trading_days)
portfolio_spy_return_df = pd.concat([period_return_df, period_return_bond_df, spy_period_return_df], axis=1)
portfolio_spy_return_df.columns = ['ETF Rotation','ETF Rotation (bond)', 'SPY']
portfolio_spy_return_df = round(portfolio_spy_return_df * 100, 2)

print(tabulate(portfolio_spy_return_df, headers=[*portfolio_spy_return_df.columns], tablefmt='fancy_grid'))

average_return_df = pd.DataFrame(portfolio_spy_return_df.mean()).transpose()

print(tabulate(average_return_df, headers=[*average_return_df.columns], tablefmt='fancy_grid'))

five_year_start = datetime.fromisoformat('2017-01-03')
three_year_start = datetime.fromisoformat('2019-01-02')
# end_date is today minus one day

five_year_portfolio_df, assets_df = portfolio_return(holdings=holdings,
                                              asset_percent=equity_percent,
                                              bond_percent=bond_percent,
                                              asset_etfs=asset_adj_close,
                                              bond_etfs=fixed_income_adjclose,
                                              start_date=five_year_start,
                                              end_date=end_date)

plot_df = build_plot_data(holdings, five_year_portfolio_df, spy_df)

three_year_portfolio_df, assets_df = portfolio_return(holdings=holdings,
                                              asset_percent=equity_percent,
                                              bond_percent=bond_percent,
                                              asset_etfs=asset_adj_close,
                                              bond_etfs=fixed_income_adjclose,
                                              start_date=three_year_start,
                                              end_date=end_date)

plot_df = build_plot_data(holdings, three_year_portfolio_df, spy_df)

d2019_portfolio_df, d2019_assets_df = portfolio_return(holdings=holdings,
                                              asset_percent=equity_percent,
                                              bond_percent=bond_percent,
                                              asset_etfs=asset_adj_close,
                                              bond_etfs=fixed_income_adjclose,
                                              start_date=d2019_start,
                                              end_date=d2019_end)

plot_df = build_plot_data(holdings, d2019_portfolio_df, spy_df)
plot_df.plot(grid=True, title=f'2019: {d2019_start.strftime("%m/%d/%Y")} - {d2019_end.strftime("%m/%d/%Y")} and SPY', figsize=(10,6))

spy_unadj = pd.DataFrame(asset_adj_close['SPY'])

spyonly_df, t = portfolio_return(holdings=holdings,
                                              asset_percent=equity_percent,
                                              bond_percent=bond_percent,
                                              asset_etfs=spy_unadj,
                                              bond_etfs=spy_unadj,
                                              start_date=d2019_start,
                                              end_date=d2019_end)

plot_df = build_plot_data(holdings, spyonly_df, spy_unadj)


def get_trendlineprofits_data(path: str) -> pd.DataFrame:
    """
    Return a composite DataFrame that is composed of period data built from the price time series
    and the Trendline Profits allocation data.  Here is an example:

       start_ix  start_date  end_ix    end_date equity bond
0      2789  2019-01-02    2809  2019-01-31    SHY  TLT
1      2810  2019-02-01    2828  2019-02-28    QQQ  TLT
2      2829  2019-03-01    2849  2019-03-29    QQQ  TLT
3      2850  2019-04-01    2870  2019-04-30    QQQ  JNK
4      2871  2019-05-01    2892  2019-05-31    QQQ  JNK
    """
    def read_trendlineprofits_data(file_path: str) -> pd.DataFrame:
        temp_file_path = Path(file_path)
        file_size = 0
        allocation_df = pd.DataFrame()
        if temp_file_path.exists():
            file_size = temp_file_path.stat().st_size
            if file_size > 0:
                allocation_df = pd.read_csv(file_path, index_col=False)
            else:
                print(f'Error: file at {file_path} has zero size')
        else:
            print(f'Error: Could not find file at location {file_path}')

        return allocation_df


    allocations_df = read_trendlineprofits_data(path)
    start_date_col = allocations_df.columns[0] # period start date
    end_date_col = allocations_df.columns[1]   # period end date
    asset_col = allocations_df.columns[2]     # equity ETF or SHY
    bond_col = allocations_df.columns[3]      # bond ETF
    num_rows = allocations_df.shape[0]
    first_row = allocations_df[:][0:1]
    last_row = allocations_df[:][num_rows - 1:num_rows]
    trend_start_date = first_row[start_date_col].values[0]
    trend_end_date = last_row[end_date_col].values[0]
    month_periods = find_month_periods(trend_start_date, trend_end_date, asset_adj_close)
    assert month_periods.shape[0] == num_rows
    composite_df = pd.concat([month_periods, allocations_df[asset_col], allocations_df[bond_col]], axis=1)
    return composite_df


trendlineprofits_data_path = 'data/trendlineprofits_assets.csv'
composite_df = get_trendlineprofits_data(trendlineprofits_data_path)
num_rows = composite_df.shape[0]
first_row = composite_df[:][0:1]
last_row = composite_df[:][num_rows - 1:num_rows]
trend_start_date = first_row['start_date'].values[0]
trend_end_date = last_row['end_date'].values[0]

trendline_portfolio_df = portfolio_replay(holdings=holdings,
                                          asset_percent=equity_percent,
                                          bond_percent=bond_percent,
                                          asset_etfs=asset_adj_close,
                                          bond_etfs=fixed_income_adjclose,
                                          etf_selection=composite_df)

notebook_portfolio_df, assets = portfolio_return(holdings=holdings,
                                              asset_percent=equity_percent,
                                              bond_percent=bond_percent,
                                              asset_etfs=asset_adj_close,
                                              bond_etfs=fixed_income_adjclose,
                                              start_date=trend_start_date,
                                              end_date=trend_end_date)

trendline_portfolio_df.columns = ['trendline']
notebook_portfolio_df.columns = ['notebook']
plot_df = build_plot_data(holdings, notebook_portfolio_df, spy_df)
plot_df = pd.concat([plot_df, trendline_portfolio_df], axis=1)


trendline_assets = ['SHY', # 1
                    'QQQ', # 2
                    'QQQ', # 3
                    'QQQ', # 4
                    'QQQ', # 5
                    'QQQ', # 6
                    'SHY', # 7
                    'SPY', # 8
                    'QQQ', # 9
                    'SPY', # 10
                    'QQQ', # 11
                    'QQQ' # 12
                    ]

trendline_df = pd.DataFrame(trendline_assets)
trendline_df.index = d2019_assets_df.index
d2019_assets_df['trendline'] = trendline_df
d2019_assets_df.drop(d2019_assets_df.columns[1], inplace=True, axis=1)
print(tabulate(d2019_assets_df, headers=['Equity ETFs', 'Trendline ETFs'], tablefmt='fancy_grid'))


def display_asset_selection(month_start: str, prices_df: pd.DataFrame) -> None:
    look_back_end = datetime.fromisoformat(month_start)
    look_back_start = look_back_end - relativedelta(months=3)
    date_index = prices_df.index
    start_ix = findDateIndex(date_index, look_back_start)
    end_ix = findDateIndex(date_index, look_back_end)
    name = chooseAssetName(start_ix, end_ix, prices_df)
    period_return_df = percent_return_df(start_date=look_back_start,
                                         end_date=look_back_end,
                                         prices_df=prices_df)
    title_str =f'{look_back_start.strftime("%m/%d/%Y")} - {look_back_end.strftime("%m/%d/%Y")} asset {name}'
    ax = period_return_df.plot(grid=True, title=title_str, figsize=(10,6))
    ax.set_ylabel('Return Percent')

fixed_income_plus_shy = pd.concat([fixed_income_adjclose, shy_adj_close], axis=1)
portfolio_bond_plus_shy_df, assets_df = portfolio_return(holdings=holdings,
                                              asset_percent=equity_percent,
                                              bond_percent=bond_percent,
                                              asset_etfs=asset_adj_close,
                                              bond_etfs=fixed_income_plus_shy,
                                              start_date=start_date,
                                              end_date=end_date)

plot_df = build_plot_data(holdings, portfolio_bond_plus_shy_df, spy_df)


limited_asset_set_df = asset_adj_close[['SPY', 'SHY']]

portfolio_limited_df, assets_df = portfolio_return(holdings=holdings,
                                              asset_percent=equity_percent,
                                              bond_percent=bond_percent,
                                              asset_etfs=limited_asset_set_df,
                                              bond_etfs=fixed_income_adjclose,
                                              start_date=start_date,
                                              end_date=end_date)

plot_df = build_plot_data(holdings, portfolio_limited_df, spy_df)


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

portfolio_new_df, assets_df = portfolio_return(holdings=holdings,
                                              asset_percent=equity_percent,
                                              bond_percent=bond_percent,
                                              asset_etfs=new_etf_set,
                                              bond_etfs=new_bond_set,
                                              start_date=start_date,
                                              end_date=end_date)

plot_df = build_plot_data(holdings, portfolio_new_df, spy_df)

assets_plus_short = pd.concat([asset_adj_close, short_etf_adj_close], axis=1)
portfolio_plus_short_df, assets_df = portfolio_return(holdings=holdings,
                                              asset_percent=equity_percent,
                                              bond_percent=bond_percent,
                                              asset_etfs=assets_plus_short,
                                              bond_etfs=fixed_income_adjclose,
                                              start_date=start_date,
                                              end_date=end_date)

plot_df = build_plot_data(holdings, portfolio_plus_short_df, spy_df)

def calculate_return_series(close_prices_df: pd.DataFrame,
                            start_date: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    :param close_prices_df: a DataFrame of close prices where all of the data aligns on date
    :param start_date: the start date to use in calculating the returns
    :return: a Series with the past three month return and the return for the next month
    """
    date_index = close_prices_df.index
    end_date = date_index[-1]
    end_date_t = convert_date(end_date)
    back_delta = relativedelta(months=3)
    forward_delta = relativedelta(months=1)
    start_date_i = start_date
    three_month_return_df = pd.DataFrame()
    one_month_return_df = pd.DataFrame()
    while start_date_i <= end_date_t:
        # Start of the back-test data
        back_start = start_date_i - back_delta
        # End of the backtest data
        back_end = start_date_i
        # end of the forward data period (e.g., one month)
        forward_end = start_date_i + forward_delta
        start_ix = findDateIndex(date_index, back_start)
        end_ix = findDateIndex(date_index, back_end)
        forward_ix = findDateIndex(date_index, forward_end)
        if start_ix >= 0 and end_ix >= 0 and forward_ix >= 0:
            three_month_a = (close_prices_df[:][start_ix:start_ix+1].values / close_prices_df[:][end_ix:end_ix+1].values) - 1
            three_month_return_df = pd.concat([three_month_return_df, pd.DataFrame(three_month_a)])
            one_month_a = (close_prices_df[:][forward_ix:forward_ix+1].values / close_prices_df[:][end_ix:end_ix+1].values) - 1
            one_month_return_df = pd.concat([one_month_return_df, pd.DataFrame(one_month_a)])
        else:
            break
        start_date_i = forward_end
    three_month_return_df.columns = close_prices_df.columns
    one_month_return_df.columns = close_prices_df.columns
    return three_month_return_df, one_month_return_df


all_etf_adj_close = pd.concat([equity_adj_close,
                               new_equity_adj_close,
                               short_etf_adj_close,
                               shy_adj_close], axis=1)
corr_end_date = start_date + relativedelta(years=8)
date_index = all_etf_adj_close.index
corr_end_ix = findDateIndex(date_index, corr_end_date)
all_etf_adj_close_trunc = all_etf_adj_close[:][0:corr_end_ix+1]
three_month_df, one_month_df = calculate_return_series(close_prices_df=all_etf_adj_close_trunc, start_date=start_date)
return_corr = three_month_df.corrwith(one_month_df)
return_corr.sort_values(ascending=False, inplace=True)
return_corr_df = pd.DataFrame(return_corr)

print(tabulate(return_corr_df, headers=['Correlation'], tablefmt='fancy_grid'))

etf_corr_set = return_corr_df[:][return_corr_df >= 0.10].dropna()
high_corr_etfs = all_etf_adj_close[etf_corr_set.index]
# make sure that SHY is included in the ETF set
if not 'SHY' in high_corr_etfs.columns:
    high_corr_etfs = pd.concat([high_corr_etfs, shy_adj_close], axis=1)

high_corr_portfolio_df, assets_df = portfolio_return(holdings=holdings,
                                              asset_percent=equity_percent,
                                              bond_percent=bond_percent,
                                              asset_etfs=high_corr_etfs,
                                              bond_etfs=fixed_income_adjclose,
                                              start_date=corr_end_date,
                                              end_date=end_date)

twelve_percent_df,  assets_df = portfolio_return(holdings=holdings,
                                              asset_percent=equity_percent,
                                              bond_percent=bond_percent,
                                              asset_etfs=asset_adj_close,
                                              bond_etfs=fixed_income_adjclose,
                                              start_date=corr_end_date,
                                              end_date=end_date)


spy_df_adj, t = adjust_time_series(spy_df, high_corr_portfolio_df)

plot_df = build_plot_data(holdings, high_corr_portfolio_df, spy_df_adj)
plot_df['twelve percent'] = twelve_percent_df
plot_df.columns = ['Correlation', 'SPY', 'twelve percent']

return_corr_a = return_corr.values
# https://stats.stackexchange.com/questions/73621/standard-error-from-correlation-coefficient
n = three_month_df.shape[0]
corr_se_1 = sqrt((1 - return_corr_a**2)/(n-2))
corr_se_2 = (1 - return_corr_a**2)/sqrt(n-2)

print(f'Correlation standard error: mean, equation 1 = {round(corr_se_1.mean(), 4)}, mean equation 2 = {round(corr_se_2.mean(), 4)}')


three_month_df, one_month_df = calculate_return_series(close_prices_df=all_etf_adj_close, start_date=start_date)
return_corr = three_month_df.corrwith(one_month_df)
return_corr.sort_values(ascending=False, inplace=True)
return_corr_df = pd.DataFrame(return_corr)

return_corr_a = return_corr.values
# https://stats.stackexchange.com/questions/73621/standard-error-from-correlation-coefficient
n = three_month_df.shape[0]
corr_se_1 = sqrt((1 - return_corr_a**2)/(n-2))
corr_se_2 = (1 - return_corr_a**2)/sqrt(n-2)

print(tabulate(return_corr_df, headers=['Correlation'], tablefmt='fancy_grid'))

print('\n')
print(f'Number of returns used to calculate the correlations: {n}')

print(f'Correlation standard error: mean equation 1 = {round(corr_se_1.mean(), 4)}, mean equation 2 = {round(corr_se_2.mean(), 4)}')




print("Hi there")