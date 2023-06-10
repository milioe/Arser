import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import datetime
import matplotlib.pyplot as plt
import itertools
import altair as alt
import pyfolio as pf

# Las listas están en el archivo 'ticker.py'
from tickers import lista_completa, lista_media, lista_corta

def MV_criterion(weights, data):
    Lambda = 1
    W = 1
    Wbar = 1 + 0.25 / 100
    portfolio_return = np.multiply(data, np.transpose(weights))
    portfolio_return = portfolio_return.sum(axis=1)
    mean = np.mean(portfolio_return, axis=0)
    std = np.std(portfolio_return, axis=0)
    criterion = Wbar ** (1 - Lambda) / (1 + Lambda) + Wbar ** (-Lambda) \
                * W * mean - Lambda / 2 * Wbar ** (-1 - Lambda) * W ** 2 * std ** 2
    criterion = -criterion
    return criterion


def obtain_mv(database, train_set):
    n = database.shape[1]
    x0 = np.ones(n)
    cons = ({'type': 'eq', 'fun': lambda x: sum(abs(x)) - 1})
    Bounds = [(0, 1) for i in range(0, n)]
    res_MV = minimize(MV_criterion, x0, method="SLSQP", args=(train_set), bounds=Bounds, constraints=cons, options={'disp': True})
    X_MV = res_MV.x
    return X_MV

# Función para descargar datos
def download_data(tickers):
    data = yf.download(tickers, start=datetime.datetime.now() - datetime.timedelta(days=365))['Adj Close']
    return data

def calculate_annual_returns(weights, returns):
    """
    Calculate the annualized return of a portfolio given the weights and daily returns of the ETFs in the portfolio.

    :param weights: List of weights of the ETFs in the portfolio.
    :param returns: DataFrame of daily returns of the ETFs.
    :return: Annualized return of the portfolio.
    """
    portfolio_return_daily = (weights * returns).sum(axis=1)
    portfolio_return_annualized = ((1 + portfolio_return_daily.mean()) ** 252) - 1
    return portfolio_return_annualized


def calculate_volatility(portfolio_returns, weights):
    cov_matrix = portfolio_returns.cov()
    weights = pd.Series(weights, index=portfolio_returns.columns)
    portfolio_volatility = np.sqrt(np.dot(np.dot(weights.T, cov_matrix), weights))
    return cov_matrix, portfolio_volatility



def calculate_sharpe_ratios(annualized_portfolio_return, annualized_benchmark_return, portfolio_std_deviation):
    """
    Calculate the Sharpe Ratio of a portfolio.

    :param annualized_portfolio_return: The annualized return of the portfolio.
    :param annualized_benchmark_return: The annualized return of the benchmark.
    :param portfolio_std_deviation: The standard deviation (volatility) of the portfolio.
    :return: The Sharpe Ratio of the portfolio.
    """
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
    """
    Get the latest available price for a given ticker.

    :param ticker: The ticker symbol to get the price for.
    :return: The latest available price.
    """
    data = yf.download(ticker, start=datetime.datetime.now() - datetime.timedelta(days=365))['Adj Close']
    # Buscar hacia atrás en los datos hasta que encontremos un valor no nulo
    for price in reversed(data):
        if not np.isnan(price):
            return price


# Configuración de Streamlit
st.set_page_config(layout="wide")
st.title('Optimizador de Portafolio')

# Mapeo de listas
etf_lists_map = {"Lista 1": lista_completa, "Lista 2": lista_media, "Lista 3": lista_corta}

# Widget placeholder
a = st.empty()

# Lista de optimizadores ficticia
optimizers = ["Mean-Variance", "Optimizer 2", "Optimizer 3", "Optimizer 4", "Optimizer 5"]

# Lista de benchmarks ficticia
benchmarks = ["^GSPC", "Benchmark 2", "Benchmark 3"]

# Combinaciones de ETF
etf_combinations = list(range(2, 11))  # De 2 a 10, modifícalo como necesites

col1, col2 = a.columns(2)

with col1:
    col11, col22 = st.columns(2) 
    with col11:
        amount = st.number_input('Introduce un monto', min_value=500.00)
    with col22:
        currency = st.radio("Currency", ('USD', 'MXN'))

    etf_list_name = st.radio('Elige una lista de ETFs', list(etf_lists_map.keys()))
    etf_list = etf_lists_map[etf_list_name]

    selected_optimizers = st.multiselect('Elige los optimizadores', optimizers, default=["Mean-Variance"])

with col2:
    benchmark = st.selectbox('Elige un benchmark', benchmarks, index=0)

    etf_combination = st.multiselect('Elige las combinaciones de ETFs en cada portafolio', etf_combinations, default=2)

button = st.empty()
if button.button('Optimizar'):
    button.empty()
    a.empty()
    
    tab = st.tabs(["Info", "Portfolio"])

    
    with tab[0]:
        st.header("Gráfica de datos")
        data = download_data(etf_list)
        st.line_chart(data)
        

    with tab[1]:
        header = st.header("Portfolio")
        weights_df = pd.DataFrame(columns=etf_list)
        returns_df = pd.DataFrame(columns=["Portfolio", "Annual Return", "Volatility", "Sharpe Ratio", "Standard Deviation", "Sortino Ratio", "Upside Capture Ratio", "Downside Capture Ratio", "Capture Ratio"])
        data_returns = data.pct_change().dropna()
        portfolio_number = 1
        # Descargar y calcular los rendimientos del benchmark
        benchmark_data = yf.download(benchmark, start=datetime.datetime.now() - datetime.timedelta(days=365))['Adj Close']
        benchmark_returns = benchmark_data.pct_change().dropna()
        annualized_benchmark_return = np.mean(benchmark_returns) * 252
        

        for n in etf_combination:
            for subset in itertools.combinations(etf_list, n):
                weights = obtain_mv(data[list(subset)], data[list(subset)])
                weights_dict = dict(zip(subset, weights))
                # Agrega los pesos a weights_df
                weights_df = pd.concat([weights_df, pd.DataFrame([weights_dict])], ignore_index=True)
                annual_return = calculate_annual_returns(weights, data_returns[list(subset)])
                portfolio_returns = np.dot(weights, data_returns[list(subset)].T)

                # Calcula la volatilidad del portafolio
                cov_matrix, portfolio_volatility = calculate_volatility(data_returns[list(subset)], weights)
                # Calcula el Sharpe ratio
                sharpe_ratio = calculate_sharpe_ratios(annual_return, annualized_benchmark_return, portfolio_volatility)
                # Calcula la desviación estándar del portafolio
                portfolio_std_dev = calculate_std_dev(weights, cov_matrix)
                # Calcula el Sortino ratio
                sortino = sortino_ratio(weights, data_returns[list(subset)], annual_return, annualized_benchmark_return)
                # Calcula los ratios de captura al alza y a la baja
                up_capture = upside_capture_ratio(data_returns[list(subset)], benchmark_returns)
                down_capture = downside_capture_ratio(data_returns[list(subset)], benchmark_returns)
                # Calcula el Capture Ratio
                capture_ratio = up_capture / down_capture
                
                # Agrega la información al dataframe de rendimientos
                returns_df = pd.concat([returns_df, pd.DataFrame([{"Portfolio": portfolio_number, "Annual Return": annual_return, "Volatility": portfolio_volatility, "Sharpe Ratio": sharpe_ratio, "Standard Deviation": portfolio_std_dev, "Sortino Ratio": sortino, "Upside Capture Ratio": up_capture, "Downside Capture Ratio": down_capture, "Capture Ratio": capture_ratio}])], ignore_index=True)

                portfolio_number += 1

                # Calcular el costo del portfolio aquí
                for idx, row in weights_df.iterrows():
                    portfolio_cost = 0
                    for ticker, weight in row.items():
                        if not np.isnan(weight):
                            # Buscar el último precio en el dataframe data en lugar de hacer una nueva llamada a la API
                            latest_price = data[ticker].iloc[-1]
                            portfolio_cost += weight * latest_price
                    returns_df.loc[idx, "Portfolio Cost"] = portfolio_cost
                        
        # st.write(weights_df)
        subtabs = st.tabs(["Returns", "Volatility", "Sharpe Ratio", "Capture Ratio"])

        # Dentro de cada subtab
        with subtabs[0]:
            top_5_etfs_returns = returns_df.sort_values(by="Annual Return", ascending=False).head(5)
            # st.write(top_5_etfs_returns)
            for idx, row in top_5_etfs_returns.iterrows():
                weights_row = weights_df.loc[idx]
                data_for_chart = []

                # Utilizar la función expander para cada portafolio
                with st.expander(f"Portfolio {idx}"):
                    subsubtabs = st.tabs(["Weights", "Matrix", "Backtesting"])

                    with subsubtabs[0]:
                        # Crear las columnas
                        col1, col2 = st.columns(2)

                        for ticker, weight in weights_row.items():
                            if not np.isnan(weight):
                                latest_price = data[ticker].iloc[-1]
                                percent_change = data[ticker].pct_change().iloc[-1]
                                # Utilizar la primera columna para los indicadores
                                col1.metric(label=ticker, value="${:.2f}".format(latest_price), delta="{:.2%}".format(percent_change))

                                # Calcular el número de acciones que se pueden comprar con la cantidad dada y el último precio del ETF
                                shares = int((amount * weight) / latest_price)
                                data_for_chart.append({"ETF": ticker, "Weight": weight, "Shares": shares})

                        source = pd.DataFrame(data_for_chart)
                        chart = (
                            alt.Chart(source)
                            .mark_bar()
                            .encode(
                                x='ETF:N',
                                y='Weight:Q',
                                tooltip=['ETF', 'Weight', 'Shares'],
                                color=alt.Color('ETF:N', legend=None)
                            )
                            .properties(
                                title='Porcentaje de compra por ETF',
                                width=600,
                                height=400,
                            )
                        )
                        # Utilizar la segunda columna para el gráfico
                        col2.altair_chart(chart, use_container_width=True)

                        # Añadir un dataframe transpuesto con las métricas del portafolio debajo de las dos columnas
                        portfolio_metrics = returns_df.loc[idx, ["Annual Return", "Volatility", "Sharpe Ratio", "Standard Deviation", "Sortino Ratio", "Upside Capture Ratio", "Downside Capture Ratio", "Capture Ratio"]].to_frame().T
                        st.dataframe(portfolio_metrics)

                    
                    with subsubtabs[1]:
                        colmat1, colmat2 = st.columns(2)

                        with colmat1:
                            # Calcular la matriz de correlación para los activos en la cartera
                            portfolio_returns = data_returns[weights_row.dropna().index]
                            correlation_matrix = portfolio_returns.corr()

                            # Transformar la matriz de correlación en formato largo
                            correlation_matrix_long = correlation_matrix.stack().reset_index()
                            correlation_matrix_long.columns = ['variable', 'variable2', 'correlation']

                            # Generar el gráfico con Altair
                            base = alt.Chart(correlation_matrix_long).encode(
                                x='variable:O',
                                y='variable2:O'
                            ).properties(
                                width=600,  # Establecer el ancho del gráfico
                                height=500  # Establecer la altura del gráfico
                            )
                            
                            text = base.mark_text(fontSize=20).encode(  # Aumentar el tamaño del texto
                                text=alt.Text('correlation:Q', format='.2f'),
                                color=alt.condition(
                                    alt.datum.correlation > 0.5, 
                                    alt.value('white'),
                                    alt.value('black')
                                )
                            )

                            cor_plot = base.mark_rect().encode(
                                color=alt.Color('correlation:Q', scale=alt.Scale(domain=[-1, 0, 1], range=["lightblue", "white", "darkblue"]))
                            )

                            colmat1.altair_chart(cor_plot + text, use_container_width=True)

                        with colmat2:
                            # Calculate the covariance matrix for the assets in the portfolio
                            covariance_matrix = portfolio_returns.cov()

                            # Transform the covariance matrix to long format
                            covariance_matrix_long = covariance_matrix.stack().reset_index()
                            covariance_matrix_long.columns = ['variable', 'variable2', 'covariance']

                            # Generate the chart with Altair
                            base = alt.Chart(covariance_matrix_long).encode(
                                x='variable:O',
                                y='variable2:O'
                            ).properties(
                                width=600,  # Set the width of the chart
                                height=500  # Set the height of the chart
                            )
                            
                            text = base.mark_text(fontSize=20).encode(  # Increase the size of the text
                                text=alt.Text('covariance:Q', format='.2f'),
                                color=alt.condition(
                                    alt.datum.covariance > 0.5, 
                                    alt.value('white'),
                                    alt.value('black')
                                )
                            )

                            cov_plot = base.mark_rect().encode(
                                color=alt.Color('covariance:Q', scale=alt.Scale(domain=[-1, 0, 1], range=["lightblue", "white", "darkblue"]))
                            )

                            colmat2.altair_chart(cov_plot + text, use_container_width=True)

                    with subsubtabs[2]:
                        col1, col2 = st.columns(2)
                        # Crear un backtest de la cartera en función de los pesos y el dataframe de datos
                        portfolio_weights = weights_row.dropna()

                        # Calcula el retorno acumulativo de la cartera a lo largo del tiempo
                        portfolio_returns = (data_returns[portfolio_weights.index] * portfolio_weights).sum(axis=1)
                        cumulative_returns = (1 + portfolio_returns).cumprod() - 1

                        # Calcula los retornos acumulados del benchmark
                        benchmark_cumulative_returns = (1 + benchmark_returns).cumprod() - 1

                        with col1:
                            # Combinar las series de retornos acumulados en un dataframe
                            combined_returns = pd.DataFrame({
                                'Portfolio': cumulative_returns,
                                'Benchmark': benchmark_cumulative_returns
                            })

                            # Graficar los retornos acumulados
                            st.line_chart(combined_returns)

                        with col2:
                            # Crear un gráfico de área que muestre la contribución de cada activo a la cartera en el tiempo
                            individual_cumulative_returns = (1 + (portfolio_weights * data_returns[portfolio_weights.index])).cumprod() - 1
                            st.area_chart(individual_cumulative_returns)
        ##############################################################################################################################################
        # Dentro de cada subtab
        with subtabs[1]:
            top_5_etfs_vol = returns_df.sort_values(by="Volatility", ascending=True).head(5)
            # st.write(top_5_etfs_vol)
            for idx, row in top_5_etfs_vol.iterrows():
                weights_row = weights_df.loc[idx]
                data_for_chart = []

                # Utilizar la función expander para cada portafolio
                with st.expander(f"Portfolio {idx}"):
                    subsubtabs = st.tabs(["Weights", "Matrix", "Backtesting"])

                    with subsubtabs[0]:
                        # Crear las columnas
                        col1, col2 = st.columns(2)

                        for ticker, weight in weights_row.items():
                            if not np.isnan(weight):
                                latest_price = data[ticker].iloc[-1]
                                percent_change = data[ticker].pct_change().iloc[-1]
                                # Utilizar la primera columna para los indicadores
                                col1.metric(label=ticker, value="${:.2f}".format(latest_price), delta="{:.2%}".format(percent_change))

                                # Calcular el número de acciones que se pueden comprar con la cantidad dada y el último precio del ETF
                                shares = int((amount * weight) / latest_price)
                                data_for_chart.append({"ETF": ticker, "Weight": weight, "Shares": shares})

                        source = pd.DataFrame(data_for_chart)
                        chart = (
                            alt.Chart(source)
                            .mark_bar()
                            .encode(
                                x='ETF:N',
                                y='Weight:Q',
                                tooltip=['ETF', 'Weight', 'Shares'],
                                color=alt.Color('ETF:N', legend=None)
                            )
                            .properties(
                                title='Porcentaje de compra por ETF',
                                width=600,
                                height=400,
                            )
                        )
                        # Utilizar la segunda columna para el gráfico
                        col2.altair_chart(chart, use_container_width=True)

                        # Añadir un dataframe transpuesto con las métricas del portafolio debajo de las dos columnas
                        portfolio_metrics = returns_df.loc[idx, ["Annual Return", "Volatility", "Sharpe Ratio", "Standard Deviation", "Sortino Ratio", "Upside Capture Ratio", "Downside Capture Ratio", "Capture Ratio"]].to_frame().T
                        st.dataframe(portfolio_metrics)

                    
                    with subsubtabs[1]:
                        colmat1, colmat2 = st.columns(2)

                        with colmat1:
                            # Calcular la matriz de correlación para los activos en la cartera
                            portfolio_returns = data_returns[weights_row.dropna().index]
                            correlation_matrix = portfolio_returns.corr()

                            # Transformar la matriz de correlación en formato largo
                            correlation_matrix_long = correlation_matrix.stack().reset_index()
                            correlation_matrix_long.columns = ['variable', 'variable2', 'correlation']

                            # Generar el gráfico con Altair
                            base = alt.Chart(correlation_matrix_long).encode(
                                x='variable:O',
                                y='variable2:O'
                            ).properties(
                                width=600,  # Establecer el ancho del gráfico
                                height=500  # Establecer la altura del gráfico
                            )
                            
                            text = base.mark_text(fontSize=20).encode(  # Aumentar el tamaño del texto
                                text=alt.Text('correlation:Q', format='.2f'),
                                color=alt.condition(
                                    alt.datum.correlation > 0.5, 
                                    alt.value('white'),
                                    alt.value('black')
                                )
                            )

                            cor_plot = base.mark_rect().encode(
                                color=alt.Color('correlation:Q', scale=alt.Scale(domain=[-1, 0, 1], range=["lightblue", "white", "darkblue"]))
                            )

                            colmat1.altair_chart(cor_plot + text, use_container_width=True)

                        with colmat2:
                            # Calculate the covariance matrix for the assets in the portfolio
                            covariance_matrix = portfolio_returns.cov()

                            # Transform the covariance matrix to long format
                            covariance_matrix_long = covariance_matrix.stack().reset_index()
                            covariance_matrix_long.columns = ['variable', 'variable2', 'covariance']

                            # Generate the chart with Altair
                            base = alt.Chart(covariance_matrix_long).encode(
                                x='variable:O',
                                y='variable2:O'
                            ).properties(
                                width=600,  # Set the width of the chart
                                height=500  # Set the height of the chart
                            )
                            
                            text = base.mark_text(fontSize=20).encode(  # Increase the size of the text
                                text=alt.Text('covariance:Q', format='.2f'),
                                color=alt.condition(
                                    alt.datum.covariance > 0.5, 
                                    alt.value('white'),
                                    alt.value('black')
                                )
                            )

                            cov_plot = base.mark_rect().encode(
                                color=alt.Color('covariance:Q', scale=alt.Scale(domain=[-1, 0, 1], range=["lightblue", "white", "darkblue"]))
                            )

                            colmat2.altair_chart(cov_plot + text, use_container_width=True)

                    with subsubtabs[2]:
                        col1, col2 = st.columns(2)
                        # Crear un backtest de la cartera en función de los pesos y el dataframe de datos
                        portfolio_weights = weights_row.dropna()

                        # Calcula el retorno acumulativo de la cartera a lo largo del tiempo
                        portfolio_returns = (data_returns[portfolio_weights.index] * portfolio_weights).sum(axis=1)
                        cumulative_returns = (1 + portfolio_returns).cumprod() - 1

                        # Calcula los retornos acumulados del benchmark
                        benchmark_cumulative_returns = (1 + benchmark_returns).cumprod() - 1

                        with col1:
                            # Combinar las series de retornos acumulados en un dataframe
                            combined_returns = pd.DataFrame({
                                'Portfolio': cumulative_returns,
                                'Benchmark': benchmark_cumulative_returns
                            })

                            # Graficar los retornos acumulados
                            st.line_chart(combined_returns)

                        with col2:
                            # Crear un gráfico de área que muestre la contribución de cada activo a la cartera en el tiempo
                            individual_cumulative_returns = (1 + (portfolio_weights * data_returns[portfolio_weights.index])).cumprod() - 1
                            st.area_chart(individual_cumulative_returns)
        ############################################################################################################################################################
        
        with subtabs[2]:
            top_5_etfs_sharpe = returns_df.sort_values(by="Sharpe Ratio", ascending=False).head(5)
            # st.write(top_5_etfs_sharpe)
            for idx, row in top_5_etfs_sharpe.iterrows():
                weights_row = weights_df.loc[idx]
                data_for_chart = []

                # Utilizar la función expander para cada portafolio
                with st.expander(f"Portfolio {idx}"):
                    subsubtabs = st.tabs(["Weights", "Matrix", "Backtesting"])

                    with subsubtabs[0]:
                        # Crear las columnas
                        col1, col2 = st.columns(2)

                        for ticker, weight in weights_row.items():
                            if not np.isnan(weight):
                                latest_price = data[ticker].iloc[-1]
                                percent_change = data[ticker].pct_change().iloc[-1]
                                # Utilizar la primera columna para los indicadores
                                col1.metric(label=ticker, value="${:.2f}".format(latest_price), delta="{:.2%}".format(percent_change))

                                # Calcular el número de acciones que se pueden comprar con la cantidad dada y el último precio del ETF
                                shares = int((amount * weight) / latest_price)
                                data_for_chart.append({"ETF": ticker, "Weight": weight, "Shares": shares})

                        source = pd.DataFrame(data_for_chart)
                        chart = (
                            alt.Chart(source)
                            .mark_bar()
                            .encode(
                                x='ETF:N',
                                y='Weight:Q',
                                tooltip=['ETF', 'Weight', 'Shares'],
                                color=alt.Color('ETF:N', legend=None)
                            )
                            .properties(
                                title='Porcentaje de compra por ETF',
                                width=600,
                                height=400,
                            )
                        )
                        # Utilizar la segunda columna para el gráfico
                        col2.altair_chart(chart, use_container_width=True)

                        # Añadir un dataframe transpuesto con las métricas del portafolio debajo de las dos columnas
                        portfolio_metrics = returns_df.loc[idx, ["Annual Return", "Volatility", "Sharpe Ratio", "Standard Deviation", "Sortino Ratio", "Upside Capture Ratio", "Downside Capture Ratio", "Capture Ratio"]].to_frame().T
                        st.dataframe(portfolio_metrics)

                    
                    with subsubtabs[1]:
                        colmat1, colmat2 = st.columns(2)

                        with colmat1:
                            # Calcular la matriz de correlación para los activos en la cartera
                            portfolio_returns = data_returns[weights_row.dropna().index]
                            correlation_matrix = portfolio_returns.corr()

                            # Transformar la matriz de correlación en formato largo
                            correlation_matrix_long = correlation_matrix.stack().reset_index()
                            correlation_matrix_long.columns = ['variable', 'variable2', 'correlation']

                            # Generar el gráfico con Altair
                            base = alt.Chart(correlation_matrix_long).encode(
                                x='variable:O',
                                y='variable2:O'
                            ).properties(
                                width=600,  # Establecer el ancho del gráfico
                                height=500  # Establecer la altura del gráfico
                            )
                            
                            text = base.mark_text(fontSize=20).encode(  # Aumentar el tamaño del texto
                                text=alt.Text('correlation:Q', format='.2f'),
                                color=alt.condition(
                                    alt.datum.correlation > 0.5, 
                                    alt.value('white'),
                                    alt.value('black')
                                )
                            )

                            cor_plot = base.mark_rect().encode(
                                color=alt.Color('correlation:Q', scale=alt.Scale(domain=[-1, 0, 1], range=["lightblue", "white", "darkblue"]))
                            )

                            colmat1.altair_chart(cor_plot + text, use_container_width=True)

                        with colmat2:
                            # Calculate the covariance matrix for the assets in the portfolio
                            covariance_matrix = portfolio_returns.cov()

                            # Transform the covariance matrix to long format
                            covariance_matrix_long = covariance_matrix.stack().reset_index()
                            covariance_matrix_long.columns = ['variable', 'variable2', 'covariance']

                            # Generate the chart with Altair
                            base = alt.Chart(covariance_matrix_long).encode(
                                x='variable:O',
                                y='variable2:O'
                            ).properties(
                                width=600,  # Set the width of the chart
                                height=500  # Set the height of the chart
                            )
                            
                            text = base.mark_text(fontSize=20).encode(  # Increase the size of the text
                                text=alt.Text('covariance:Q', format='.2f'),
                                color=alt.condition(
                                    alt.datum.covariance > 0.5, 
                                    alt.value('white'),
                                    alt.value('black')
                                )
                            )

                            cov_plot = base.mark_rect().encode(
                                color=alt.Color('covariance:Q', scale=alt.Scale(domain=[-1, 0, 1], range=["lightblue", "white", "darkblue"]))
                            )

                            colmat2.altair_chart(cov_plot + text, use_container_width=True)

                    with subsubtabs[2]:
                        col1, col2 = st.columns(2)
                        # Crear un backtest de la cartera en función de los pesos y el dataframe de datos
                        portfolio_weights = weights_row.dropna()

                        # Calcula el retorno acumulativo de la cartera a lo largo del tiempo
                        portfolio_returns = (data_returns[portfolio_weights.index] * portfolio_weights).sum(axis=1)
                        cumulative_returns = (1 + portfolio_returns).cumprod() - 1

                        # Calcula los retornos acumulados del benchmark
                        benchmark_cumulative_returns = (1 + benchmark_returns).cumprod() - 1

                        with col1:
                            # Combinar las series de retornos acumulados en un dataframe
                            combined_returns = pd.DataFrame({
                                'Portfolio': cumulative_returns,
                                'Benchmark': benchmark_cumulative_returns
                            })

                            # Graficar los retornos acumulados
                            st.line_chart(combined_returns)

                        with col2:
                            # Crear un gráfico de área que muestre la contribución de cada activo a la cartera en el tiempo
                            individual_cumulative_returns = (1 + (portfolio_weights * data_returns[portfolio_weights.index])).cumprod() - 1
                            st.area_chart(individual_cumulative_returns)
        #####################################################################################################################
        with subtabs[2]:
            top_5_etfs_capture = returns_df.sort_values(by="Sharpe Ratio", ascending=False).head(5)
            # st.write(top_5_etfs_capture)
            for idx, row in top_5_etfs_capture.iterrows():
                weights_row = weights_df.loc[idx]
                data_for_chart = []

                # Utilizar la función expander para cada portafolio
                with st.expander(f"Portfolio {idx}"):
                    subsubtabs = st.tabs(["Weights", "Matrix", "Backtesting"])

                    with subsubtabs[0]:
                        # Crear las columnas
                        col1, col2 = st.columns(2)

                        for ticker, weight in weights_row.items():
                            if not np.isnan(weight):
                                latest_price = data[ticker].iloc[-1]
                                percent_change = data[ticker].pct_change().iloc[-1]
                                # Utilizar la primera columna para los indicadores
                                col1.metric(label=ticker, value="${:.2f}".format(latest_price), delta="{:.2%}".format(percent_change))

                                # Calcular el número de acciones que se pueden comprar con la cantidad dada y el último precio del ETF
                                shares = int((amount * weight) / latest_price)
                                data_for_chart.append({"ETF": ticker, "Weight": weight, "Shares": shares})

                        source = pd.DataFrame(data_for_chart)
                        chart = (
                            alt.Chart(source)
                            .mark_bar()
                            .encode(
                                x='ETF:N',
                                y='Weight:Q',
                                tooltip=['ETF', 'Weight', 'Shares'],
                                color=alt.Color('ETF:N', legend=None)
                            )
                            .properties(
                                title='Porcentaje de compra por ETF',
                                width=600,
                                height=400,
                            )
                        )
                        # Utilizar la segunda columna para el gráfico
                        col2.altair_chart(chart, use_container_width=True)

                        # Añadir un dataframe transpuesto con las métricas del portafolio debajo de las dos columnas
                        portfolio_metrics = returns_df.loc[idx, ["Annual Return", "Volatility", "Sharpe Ratio", "Standard Deviation", "Sortino Ratio", "Upside Capture Ratio", "Downside Capture Ratio", "Capture Ratio"]].to_frame().T
                        st.dataframe(portfolio_metrics)

                    
                    with subsubtabs[1]:
                        colmat1, colmat2 = st.columns(2)

                        with colmat1:
                            # Calcular la matriz de correlación para los activos en la cartera
                            portfolio_returns = data_returns[weights_row.dropna().index]
                            correlation_matrix = portfolio_returns.corr()

                            # Transformar la matriz de correlación en formato largo
                            correlation_matrix_long = correlation_matrix.stack().reset_index()
                            correlation_matrix_long.columns = ['variable', 'variable2', 'correlation']

                            # Generar el gráfico con Altair
                            base = alt.Chart(correlation_matrix_long).encode(
                                x='variable:O',
                                y='variable2:O'
                            ).properties(
                                width=600,  # Establecer el ancho del gráfico
                                height=500  # Establecer la altura del gráfico
                            )
                            
                            text = base.mark_text(fontSize=20).encode(  # Aumentar el tamaño del texto
                                text=alt.Text('correlation:Q', format='.2f'),
                                color=alt.condition(
                                    alt.datum.correlation > 0.5, 
                                    alt.value('white'),
                                    alt.value('black')
                                )
                            )

                            cor_plot = base.mark_rect().encode(
                                color=alt.Color('correlation:Q', scale=alt.Scale(domain=[-1, 0, 1], range=["lightblue", "white", "darkblue"]))
                            )

                            colmat1.altair_chart(cor_plot + text, use_container_width=True)

                        with colmat2:
                            # Calculate the covariance matrix for the assets in the portfolio
                            covariance_matrix = portfolio_returns.cov()

                            # Transform the covariance matrix to long format
                            covariance_matrix_long = covariance_matrix.stack().reset_index()
                            covariance_matrix_long.columns = ['variable', 'variable2', 'covariance']

                            # Generate the chart with Altair
                            base = alt.Chart(covariance_matrix_long).encode(
                                x='variable:O',
                                y='variable2:O'
                            ).properties(
                                width=600,  # Set the width of the chart
                                height=500  # Set the height of the chart
                            )
                            
                            text = base.mark_text(fontSize=20).encode(  # Increase the size of the text
                                text=alt.Text('covariance:Q', format='.2f'),
                                color=alt.condition(
                                    alt.datum.covariance > 0.5, 
                                    alt.value('white'),
                                    alt.value('black')
                                )
                            )

                            cov_plot = base.mark_rect().encode(
                                color=alt.Color('covariance:Q', scale=alt.Scale(domain=[-1, 0, 1], range=["lightblue", "white", "darkblue"]))
                            )

                            colmat2.altair_chart(cov_plot + text, use_container_width=True)

                    with subsubtabs[2]:
                        col1, col2 = st.columns(2)
                        # Crear un backtest de la cartera en función de los pesos y el dataframe de datos
                        portfolio_weights = weights_row.dropna()

                        # Calcula el retorno acumulativo de la cartera a lo largo del tiempo
                        portfolio_returns = (data_returns[portfolio_weights.index] * portfolio_weights).sum(axis=1)
                        cumulative_returns = (1 + portfolio_returns).cumprod() - 1

                        # Calcula los retornos acumulados del benchmark
                        benchmark_cumulative_returns = (1 + benchmark_returns).cumprod() - 1

                        with col1:
                            # Combinar las series de retornos acumulados en un dataframe
                            combined_returns = pd.DataFrame({
                                'Portfolio': cumulative_returns,
                                'Benchmark': benchmark_cumulative_returns
                            })

                            # Graficar los retornos acumulados
                            st.line_chart(combined_returns)

                        with col2:
                            # Crear un gráfico de área que muestre la contribución de cada activo a la cartera en el tiempo
                            individual_cumulative_returns = (1 + (portfolio_weights * data_returns[portfolio_weights.index])).cumprod() - 1
                            st.area_chart(individual_cumulative_returns)




