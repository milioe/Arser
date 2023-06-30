# menu.py
import streamlit as st
# Las listas están en el archivo 'ticker.py'
from Tickers.tickers import lista_completa, lista_media, lista_corta, lista_nueva

def display_menu():

    # Mapping of lists
    etf_lists = {"List 1": lista_completa, "List 2": lista_media, "List 3": lista_corta, "List 4": lista_nueva}

    # Dummy list of optimizers
    optimizers = ["Mean-Variance", "CVaR"]

    # optimizers = ["Mean-Variance", "CVaR" "Sharpe-Ratio", "Optimización de Partículas en Enjambr PSO", "Algoritmo Genético (GA):", "Recocido Simulado (SA)", "NSGA-II", "Optimización por Colonia de Hormigas (ACO)", "Optimización Cuckoo (CO)", "Optimización por Enjambre de Luciérnagas (FA)", "Algoritmo de Abejas (BA)", "Algoritmo del Murciélago (BA)", "Algoritmo de Busqueda de Armonía (HS)", "Optimización por Fuerza Gravitacional (GSA)", "Algoritmo de Busqueda Diferencial (DE)", "Optimización por Enjambre de Abubillas (HOA)"]

    # Dummy list of benchmarks
    benchmarks = ["^GSPC", "^IXIC", "^DJI"]

    # ETF combinations
    etf_combinations = list(range(2, 11))  # From 2 to 10, adjust it as needed

    # Widget placeholder
    placeholder = st.empty()

    col1, col2 = placeholder.columns(2)

    with col1:
        sub_col1, sub_col2 = st.columns(2) 
        with sub_col1:
            amount = st.number_input('Enter an amount', min_value=500.00)
        with sub_col2:
            currency = st.radio("Currency", ('USD', 'MXN'))

        etf_list_name = st.radio('Choose an ETF list', list(etf_lists.keys()))
        selected_etf_list = etf_lists[etf_list_name]

        selected_optimizers = st.selectbox('Choose the optimizers', optimizers)

    with col2:
        benchmark = st.selectbox('Choose a benchmark', benchmarks, index=0)

        etf_combo = st.selectbox('Choose the ETF combinations for each portfolio', etf_combinations)

    button = st.empty()

    if button.button('Optimize'):
        button.empty()
        placeholder.empty()

        # Return selected_etf_list also
        return amount, currency, etf_list_name, selected_etf_list, selected_optimizers, benchmark, etf_combo, True

    # Return selected_etf_list also
    return amount, currency, etf_list_name, selected_etf_list, selected_optimizers, benchmark, etf_combo, False