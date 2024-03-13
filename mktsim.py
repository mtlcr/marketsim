
import datetime as dt
import os

import numpy as np

import pandas as pd
from util import get_data, plot_data


def author():
    return ''


def compute_portvals(
        orders_file="./orders/orders-01.csv",
        start_val=1000000,
        commission=9.95,
        impact=0.005,
):
    """
    Computes the portfolio values.

    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here
    # Read order file
    order_df = pd.read_csv(orders_file)
    # print('order_df:','\n')
    # print(order_df)
    # print('order_df dtype:','\n')
    # print(order_df.dtypes)

    # Convert column Date from Object type to datetime type
    order_df['Date'] = pd.to_datetime(order_df['Date'], format='%Y-%m-%d')
    # print('order_df dtype after converting date:','\n')
    # print(order_df.dtypes)
    # print('Start Date\n', order_df["Date"][0])
    # print('End Date\n', order_df["Date"].iloc[-1])
    # print('All symbols: \n', order_df['Symbol'].unique())

    # Get start date, end date, and all symbols in order file
    start_date = order_df["Date"][0]
    end_date = order_df["Date"].iloc[-1]
    all_symbols = order_df['Symbol'].unique()

    # Create price dataframe using start date, end date, and all symbols in order file
    price_df = get_data(order_df['Symbol'].unique(), pd.date_range(start_date, end_date))
    # print('price_df: \n', price_df)

    # remove SPY
    price_df = price_df[all_symbols]
    # print('price_df remove SPY: \n', price_df)

    # Add cash symbol $1 per symbol
    price_df['Cash'] = 1
    print('\nprice_df added cash symbol: \n', price_df)

    # Make a copy of price data frame, rename it to trade data frame and holding data frame
    trade_df = price_df.__deepcopy__()
    holding_df = price_df.__deepcopy__()
    # print('trade_df: \n', trade_df)

    # Zero out all row in trade/holding data frame
    trade_df[list(trade_df)] = 0
    holding_df[list(holding_df)] = 0
    print('\ntrade_df after zeroing out all: \n', trade_df)
    print('\nholding_df after zeroing out all: \n', holding_df)

    # remove date that are not trading dates from order df
    # price_df2 = price_df.merge(order_df, left_index=True,  right_on='Date', how='inner')
    # print('price_df2: \n', price_df2)
    # price_df = price_df.drop([price_df.index[0]])
    # print('price_df after drop index 0: \n', price_df)
    order_df = order_df.loc[order_df['Date'].isin(price_df.index)]
    print('\norder_df: \n', order_df)

    # Iterate through each order
    for index, row in order_df.iterrows():
        order_dt = order_df['Date'][index]
        symbol = row[1]
        order = row[2]
        share_abs = row[3]
        price_historical = price_df.at[order_dt, symbol]
        # print('\nindex: \n', index)
        # print('\norder_dt: \n', order_dt)
        # print('\nsymbol: \n', symbol)
        # print('\norder: \n', order)
        # print('\nshare_abs: \n', share_abs)
        # print('\nprice_historical: \n', price_historical)
        # Assign sign if BUY/SELL
        if order == 'BUY':
            share = row[3]
            price = price_df.at[order_dt, symbol] * (1 + impact)

        else:
            share = -row[3]
            price = price_df.at[order_dt, symbol] * (1 - impact)
        cost = - share * price - commission
        # print('\ncost: \n', cost)

        # Update share in trade_df for the current transaction
        trade_df.at[order_dt, symbol] += share

        # if index = 0, aka 1st order, then current cash = start_val + cash used, otherwise current cash = previous cash + cash used
        if index == 0:
            # Update cash in trade_df for the current transaction

            trade_df.at[order_dt, 'Cash'] = start_val + cost
        else:

            trade_df.at[order_dt, 'Cash'] += cost

    print('\ntrade_df \n', trade_df)

    holding_df = trade_df.cumsum()
    print('\nholding_df \n', holding_df)

    values_df = holding_df * price_df
    print('\nvalues_df \n', values_df)
    portvals = values_df.sum(axis=1)
    print('\nportvals \n', portvals)

    # start_date = dt.datetime(2008, 1, 1)
    # end_date = dt.datetime(2008, 6, 1)
    # portvals = get_data(["IBM"], pd.date_range(start_date, end_date))
    # portvals = portvals[["IBM"]]  # remove SPY
    # rv = pd.DataFrame(index=portvals.index, data=portvals.values)

    return rv
    return portvals


def get_port_stat(portvals):
    # Get portfolio stats
    daily_returns = (portvals / portvals.shift(1)) - 1  # calculate daily return of portfolio
    daily_returns.iloc[0] = 0  # Pandas leaves the 0th row full of Nans -
    daily_returns = daily_returns[1:]
    cum_ret = (portvals[-1] / portvals[0] - 1)
    avg_daily_ret = daily_returns.mean()
    std_daily_ret = daily_returns.std()
    daily_rf = 0
    k = 252
    sharpe_ratio = np.sqrt(k) * avg_daily_ret / std_daily_ret
    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio


def test_code():
    """
    Helper function to test code
    """
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    # of = "./orders/orders2.csv"
    of = "./orders/orders2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)

    # if isinstance(portvals, pd.DataFrame):
    #     portvals = portvals[portvals.columns[0]]  # just get the first column
    # else:
    "warning, code did not return a DataFrame"

    start_date = portvals.index[0]
    end_date = portvals.index[-1]

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_port_stat(portvals)

    spx = get_data(["$SPX"], pd.date_range(start_date, end_date))
    spx = spx["$SPX"]
    print('spx\n', spx)
    # repeat for $SPX.csv
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = get_port_stat(spx)

    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")


if __name__ == "__main__":
    test_code()
