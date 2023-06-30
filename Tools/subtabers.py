import numpy as np
import streamlit as st
import altair as alt
import pandas as pd


def create_chart(weights_row, amount, data):
    data_for_chart = []
    col1, col2 = st.columns(2)
    for ticker, weight in weights_row.items():
        if not np.isnan(weight):
            latest_price = data[ticker].iloc[-1]
            percent_change = data[ticker].pct_change().iloc[-1]
            col1.metric(label=ticker, value="${:.2f}".format(latest_price), delta="{:.2%}".format(percent_change))
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
    col2.altair_chart(chart, use_container_width=True)


def create_matrices(weights_row, data_returns):
    colmat1, colmat2 = st.columns(2)

    portfolio_returns = data_returns[weights_row.dropna().index]
    correlation_matrix = portfolio_returns.corr()
    correlation_matrix_long = correlation_matrix.stack().reset_index()
    correlation_matrix_long.columns = ['variable', 'variable2', 'correlation']
    plot_matrix(colmat1, correlation_matrix_long, 'correlation')

    covariance_matrix = portfolio_returns.cov()
    covariance_matrix_long = covariance_matrix.stack().reset_index()
    covariance_matrix_long.columns = ['variable', 'variable2', 'covariance']
    plot_matrix(colmat2, covariance_matrix_long, 'covariance')

def plot_matrix(column, matrix_long, matrix_type):
    base = alt.Chart(matrix_long).encode(
        x='variable:O',
        y='variable2:O'
    ).properties(width=600, height=500)
    text = base.mark_text(fontSize=20).encode(
        text=alt.Text(f'{matrix_type}:Q', format='.2f'),
        color=alt.condition(
            alt.datum[matrix_type] > 0.5, 
            alt.value('white'),
            alt.value('black')
        )
    )
    matrix_plot = base.mark_rect().encode(
        color=alt.Color(f'{matrix_type}:Q', scale=alt.Scale(domain=[-1, 0, 1], range=["lightblue", "white", "darkblue"]))
    )
    column.altair_chart(matrix_plot + text, use_container_width=True)



def create_backtesting(weights_row, data_returns, benchmark_returns):
    col1, col2 = st.columns(2)

    portfolio_weights = weights_row.dropna()
    portfolio_returns = (data_returns[portfolio_weights.index] * portfolio_weights).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    benchmark_cumulative_returns = (1 + benchmark_returns).cumprod() - 1

    with col1:
        combined_returns = pd.DataFrame({
            'Portfolio': cumulative_returns,
            'Benchmark': benchmark_cumulative_returns
        })
        st.line_chart(combined_returns)

    with col2:
        individual_cumulative_returns = (1 + (portfolio_weights * data_returns[portfolio_weights.index])).cumprod() - 1
        st.area_chart(individual_cumulative_returns)


def display_metrics(returns_row):
    returns_row_df = pd.DataFrame(returns_row).T  # Convertimos la serie en un DataFrame para presentarlo en formato de tabla
    formatted_df = returns_row_df.style.format("{:.2f}")  # Formateamos los valores como decimales con 2 puntos de precisión
    st.table(formatted_df)  # Usamos st.table para presentar la tabla formateada


def portfolio_info(returns_df, weights_df, sort_by, amount, data, data_returns, benchmark_returns):
    top_5_etfs_returns = returns_df.sort_values(by=sort_by, ascending=False).head(5)
    for idx, row in top_5_etfs_returns.iterrows():
        weights_row = weights_df.loc[idx]

        with st.expander(f"Portfolio {idx}"):
            subsubtabs = st.tabs(["Weights", "Metrics", "Matrix", "Backtesting"])

            with subsubtabs[0]:
                create_chart(weights_row, amount, data)

            with subsubtabs[1]:
                display_metrics(returns_df.loc[idx])  # Usamos la nueva función aquí

            with subsubtabs[2]:
                create_matrices(weights_row, data_returns)

            with subsubtabs[3]:
                create_backtesting(weights_row, data_returns, benchmark_returns)


