import streamlit as st
st. set_page_config(layout="wide")


def app():
    st.title('Optimizador de Portafolio')

    # Lista de ETFs ficticia
    etf_lists = ["Lista 1", "Lista 2", "Lista 3"]

    # Lista de optimizadores ficticia
    optimizers = ["Mean-Variance", "Optimizer 2", "Optimizer 3", "Optimizer 4", "Optimizer 5"]

    # Lista de benchmarks ficticia
    benchmarks = ["^GSPC", "Benchmark 2", "Benchmark 3"]

    # Combinaciones de ETF
    etf_combinations = list(range(2, 11))  # De 2 a 10, modifícalo como necesites

    st.title('Optimizador de Portafolio')

    amount = st.number_input('Introduce un monto', min_value=10000.0, step=0.1)

    etf_list = st.radio('Elige una lista de ETFs', etf_lists)

    selected_optimizers = st.multiselect('Elige los optimizadores', optimizers, default=["Mean-Variance"])

    benchmark = st.selectbox('Elige un benchmark', benchmarks, index=0)

    etf_combination = st.multiselect('Elige las combinaciones de ETFs en cada portafolio', etf_combinations, default=2)
