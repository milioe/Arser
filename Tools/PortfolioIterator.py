# PortfolioIterator.py
import itertools
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from Tools.utils import *


def portfolio_optimization(data, selected_etf_list, benchmark, etf_combo, columns):
    weights_df = pd.DataFrame(columns=selected_etf_list)
    returns_df = pd.DataFrame(columns=columns)
    data_returns = data.pct_change().dropna()
    portfolio_number = 1
    benchmark_data = yf.download(benchmark, start=datetime.datetime.now() - datetime.timedelta(days=365))['Adj Close']
    benchmark_returns = benchmark_data.pct_change().dropna()
    annualized_benchmark_return = np.mean(benchmark_returns) * 252

    for n in etf_combo:
        for subset in itertools.combinations(selected_etf_list, n):
            weights = obtain_mv(data[list(subset)], data[list(subset)])
            weights_dict = dict(zip(subset, weights))
            weights_df = pd.concat([weights_df, pd.DataFrame([weights_dict])], ignore_index=True)
            annual_return = calculate_annual_returns(weights, data_returns[list(subset)])
            portfolio_returns = np.dot(weights, data_returns[list(subset)].T)
            cov_matrix, portfolio_volatility = calculate_volatility(data_returns[list(subset)], weights)
            sharpe_ratio = calculate_sharpe_ratios(annual_return, annualized_benchmark_return, portfolio_volatility)
            portfolio_std_dev = calculate_std_dev(weights, cov_matrix)
            sortino = sortino_ratio(weights, data_returns[list(subset)], annual_return, annualized_benchmark_return)
            up_capture = upside_capture_ratio(data_returns[list(subset)], benchmark_returns)
            down_capture = downside_capture_ratio(data_returns[list(subset)], benchmark_returns)
            capture_ratio = up_capture / down_capture

            returns_df = pd.concat([returns_df, pd.DataFrame([{
                "Portfolio": portfolio_number, 
                "Annual Return": annual_return, 
                "Volatility": portfolio_volatility, 
                "Sharpe Ratio": sharpe_ratio, 
                "Standard Deviation": portfolio_std_dev, 
                "Sortino Ratio": sortino, 
                "Upside Capture Ratio": up_capture, 
                "Downside Capture Ratio": down_capture, 
                "Capture Ratio": capture_ratio}])], ignore_index=True)

            portfolio_number += 1

            for idx, row in weights_df.iterrows():
                portfolio_cost = 0
                for ticker, weight in row.items():
                    if not np.isnan(weight):
                        latest_price = data[ticker].iloc[-1]
                        portfolio_cost += weight * latest_price
                returns_df.loc[idx, "Portfolio Cost"] = portfolio_cost

    return returns_df, weights_df, data_returns, benchmark_returns
