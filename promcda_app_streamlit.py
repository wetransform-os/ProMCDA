import streamlit as st
import pandas as pd
from promcda.models.ProMCDA import ProMCDA

st.title("ProMCDA - Probabilistic Multi-Criteria Decision Analysis")

uploaded_file = st.file_uploader("Upload a CSV file with indicators and alternatives", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("input matrix uploaded:")
    st.dataframe(df)

    weights = st.text_input("Insert weights separated by commas")
    weights = [float(w.strip()) for w in weights.split(",")]

    polarity = st.text_input("Enter the polarities (e.g.: '+','-')")
    polarity = tuple(p.strip() for p in polarity.split(","))

    normalization = st.selectbox("Normalization method", ["minmax_01", "target", "standardized_any", "rank"])
    aggregation = st.selectbox("Aggregation method", ["weighted-sum", "geometric", "harmonic", "minimum"])

    if st.button("Run ProMCDA"):
        mcda = ProMCDA(
            input_matrix=df,
            weights=weights,
            polarity=polarity
        )
        results = mcda.run(normalization_method=normalization, aggregation_method=aggregation)

        for key, value in results.items():
            st.subheader(f"Results: {key}")
            st.dataframe(value)