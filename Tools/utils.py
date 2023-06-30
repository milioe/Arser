#utills.py
import numpy as np
from scipy.optimize import minimize
import datetime
import yfinance as yf
import pandas as pd
import cvxpy as cp



# Función para descargar datos
def download_data(tickers):
    data = yf.download(tickers, start=datetime.datetime.now() - datetime.timedelta(days=365))['Adj Close']
    return data

def calculate_annual_returns(weights, returns):
    portfolio_return_daily = (weights * returns).sum(axis=1)
    portfolio_return_annualized = ((1 + portfolio_return_daily.mean()) ** 252) - 1
    return portfolio_return_annualized


def calculate_volatility(portfolio_returns, weights):
    cov_matrix = portfolio_returns.cov()
    weights = pd.Series(weights, index=portfolio_returns.columns)
    portfolio_volatility = np.sqrt(np.dot(np.dot(weights.T, cov_matrix), weights))
    return cov_matrix, portfolio_volatility


def calculate_sharpe_ratios(annualized_portfolio_return, annualized_benchmark_return, portfolio_std_deviation):
    sharpe_ratio = (annualized_portfolio_return - annualized_benchmark_return) / portfolio_std_deviation
    return sharpe_ratio

def calculate_std_dev(weights, cov_matrix):
    portfolio_std_deviation = np.sqrt(np.dot(np.dot(weights.T, cov_matrix), weights)) * np.sqrt(252)  # Asumiendo 252 días de negociación en un año
    return portfolio_std_deviation

def sortino_ratio(portfolio_weights, portfolio_returns, annualized_portfolio_return, annualized_benchmark_return, threshold=0.10): # o con 0.05
    # Calcular los rendimientos del portafolio
    portfolio_daily_returns = np.dot(portfolio_returns, portfolio_weights)
    # Calcular los rendimientos por debajo del umbral
    below_threshold_returns = portfolio_daily_returns[portfolio_daily_returns < threshold]
    # Calcular la desviación estándar de los rendimientos por debajo del umbral (Downside Risk)
    downside_risk = np.std(below_threshold_returns)
    # Calcular el Sortino ratio
    sortino_ratio = (annualized_portfolio_return - annualized_benchmark_return) / downside_risk
    return sortino_ratio

def upside_capture_ratio(portfolio_returns, benchmark_returns):
    up_market_periods = benchmark_returns > 0
    if np.sum(up_market_periods) == 0:
        return np.nan
    up_market_dates = benchmark_returns.index[up_market_periods]
    filtered_portfolio_returns = portfolio_returns[np.isin(portfolio_returns.index, up_market_dates)]
    avg_portfolio_return_up_market = np.mean(filtered_portfolio_returns)
    avg_benchmark_return_up_market = np.mean(benchmark_returns[up_market_periods])
    up_market_capture_ratio = (avg_portfolio_return_up_market / avg_benchmark_return_up_market) * 100
    return up_market_capture_ratio

def downside_capture_ratio(portfolio_returns, benchmark_returns):
    down_market_periods = benchmark_returns < 0
    if np.sum(down_market_periods) == 0:
        return np.nan
    down_market_dates = benchmark_returns.index[down_market_periods]
    filtered_portfolio_returns = portfolio_returns[np.isin(portfolio_returns.index, down_market_dates)]
    avg_portfolio_return_down_market = np.mean(filtered_portfolio_returns)
    avg_benchmark_return_down_market = np.mean(benchmark_returns[down_market_periods])
    down_market_capture_ratio = (avg_portfolio_return_down_market / avg_benchmark_return_down_market) * 100
    return down_market_capture_ratio

def get_latest_price(ticker):
    data = yf.download(ticker, start=datetime.datetime.now() - datetime.timedelta(days=365))['Adj Close']
    # Buscar hacia atrás en los datos hasta que encontremos un valor no nulo
    for price in reversed(data):
        if not np.isnan(price):
            return price