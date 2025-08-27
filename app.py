import os
import pandas as pd
import numpy as np
import streamlit as st
from langchain_groq import ChatGroq


def generate_synthetic_data(num_days=365, start_date="2024-01-01", seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=num_days, freq="D")
    seasonal = np.sin(np.linspace(0, 3 * np.pi, num_days))
    df = pd.DataFrame({
        "Date": dates,
        "GDP_Growth": 3.0 + 0.5 * seasonal + rng.normal(0, 0.2, num_days),
        "Inflation_Rate": 2.0 + 0.3 * seasonal + rng.normal(0, 0.1, num_days),
        "Unemployment_Rate": 5.0 + 0.4 * seasonal + rng.normal(0, 0.2, num_days),
        "Interest_Rate": 4.0 + 0.3 * seasonal + rng.normal(0, 0.15, num_days),
        "Consumer_Sentiment": 100.0 + 3.0 * seasonal + rng.normal(0, 2, num_days),
        "Stock_Index": 3000.0 + np.cumsum(rng.normal(0.1, 1.5, num_days))
    })
    return df


def load_data(upload):
    if upload is not None:
        try:
            df = pd.read_csv(upload)
        except Exception:
            df = generate_synthetic_data()
    else:
        df = generate_synthetic_data()
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"])
        except Exception:
            pass
    return df


def build_system_prompt(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    summaries = []
    for col in numeric_cols:
        mean = df[col].mean()
        summaries.append(f"{col} promedio: {mean:.2f}")
    summary_text = "; ".join(summaries)
    return ("Eres un experto en economía. Responde preguntas de manera útil y concisa. "
            f"Estadísticas del conjunto de datos: {summary_text}.")


def main():
    st.set_page_config(page_title="Agente LLM de Economía", page_icon="💼")
    st.title("Agente LLM para Análisis Económico")
    st.write("Interactúa con este agente para explorar datos económicos y hacer preguntas sobre economía.")
    uploaded_file = st.file_uploader("Cargar archivo CSV (opcional)", type=["csv"])
    df = load_data(uploaded_file)
    st.subheader("Datos")
    st.dataframe(df)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        col = st.selectbox("Selecciona columna para graficar", numeric_cols)
        st.line_chart(df.set_index("Date")[col])
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.subheader("Chat con el agente")
    question = st.text_input("Pregunta sobre economía o el conjunto de datos:")
    if st.button("Enviar") and question:
        system_prompt = build_system_prompt(df)
        messages = [("system", system_prompt)] + st.session_state["history"] + [("human", question)]
        try:
            response = llm.invoke(messages)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            answer = f"Error: {exc}"
        st.session_state["history"].append(("human", question))
        st.session_state["history"].append(("ai", answer))
        st.markdown(f"**Respuesta:** {answer}")
    if st.session_state["history"]:
        with st.expander("Historial", expanded=False):
            for role, content in st.session_state["history"]:
                prefix = "Tú" if role == "human" else "Agente"
                st.markdown(f"**{prefix}:** {content}")


if __name__ == "__main__":
    main()
