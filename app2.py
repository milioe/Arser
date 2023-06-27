import streamlit as st
from menu import display_menu
from Tools.utils import *
from Tools.subtabers import portfolio_info
from Tools.PortfolioIterator import portfolio_optimization


# Streamlit setup
st.set_page_config(layout="wide")
st.title('Portfolio Optimizer')

# Call display_menu function from menu module
amount, currency, etf_list_name, selected_etf_list, selected_optimizers, benchmark, etf_combo, is_optimized = display_menu()

# If the button was pressed, display the tabs.
if is_optimized:

    tab = st.tabs(["Info", "Portfolio"])

    with tab[0]:
        st.header("Data chart")
        data = download_data(selected_etf_list)
        st.line_chart(data)
        
    with tab[1]:
        header = st.header("Portfolio")
        columns=["Portfolio", "Annual Return", "Volatility", "Sharpe Ratio", "Standard Deviation", "Sortino Ratio", "Upside Capture Ratio", "Downside Capture Ratio", "Capture Ratio"]
        returns_df, weights_df, data_returns, benchmark_returns = portfolio_optimization(data, selected_etf_list, benchmark, etf_combo, columns)
                        
        subtabs = st.tabs(["Returns", "Volatility", "Sharpe Ratio", "Capture Ratio"])

        with subtabs[0]:
            portfolio_info(returns_df, weights_df, "Annual Return", amount, data, data_returns, benchmark_returns)
        with subtabs[1]:
            portfolio_info(returns_df, weights_df, "Volatility", amount, data, data_returns, benchmark_returns)
        with subtabs[2]:
            portfolio_info(returns_df, weights_df, "Sharpe Ratio", amount, data, data_returns, benchmark_returns)
        with subtabs[3]:
            portfolio_info(returns_df, weights_df, "Capture Ratio", amount, data, data_returns, benchmark_returns)

        

        



