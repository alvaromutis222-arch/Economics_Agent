# -*- coding: utf-8 -*-
import os
import json
import textwrap
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

# LangChain / Groq
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# ----------------------------- CONFIG -----------------------------
st.set_page_config(
    page_title="EcoAgent Pro • Agente Económico",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
  --brand: #0EA5E9;
  --surface: #0B1221;
  --card: #111827;
  --muted: #94A3B8;
  --text: #E5E7EB;
}
html, body, [class*="css"]  {
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial;
}
section.main > div { padding-top: 0.6rem; }
h1, h2, h3, h4 { color: var(--text); }
.block-container { padding-top: 0.8rem; }
.card {
  background: var(--card);
  border: 1px solid #1f2937;
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: 0 8px 24px rgba(2,6,23,0.35);
}
.kpi {
  background: linear-gradient(180deg, #0b1221 0%, #0f172a 100%);
  border: 1px solid #1f2937; border-radius: 18px; padding: 14px 16px;
}
.chat { display: flex; gap: .6rem; margin-bottom: .75rem; }
.bubble {
  max-width: 80%; padding: .6rem .8rem; border-radius: 14px;
  border: 1px solid #1f2937; line-height: 1.35;
}
.bubble.user { background: #0f172a; color: var(--text); margin-left: auto; border-top-right-radius: 4px; }
.bubble.ai   { background: #0a1429; color: var(--text); margin-right: auto; border-top-left-radius: 4px; }
.role { font-size: .75rem; color: var(--muted); margin-bottom: .2rem; }
.small { font-size: .85rem; color: var(--muted); }
footer { visibility: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------- HELPERS -----------------------------
@st.cache_data(show_spinner=False)
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

def load_data(upload) -> pd.DataFrame:
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

def quick_stats(df: pd.DataFrame):
    num = df.select_dtypes(include=np.number)
    if num.empty:
        return {}
    stats = {
        "Filas": len(df),
        "Cols num": num.shape[1],
        "Nulos (%)": round((df.isna().sum().sum() / (df.shape[0]*df.shape[1]))*100, 2),
    }
    vol = num.std().sort_values(ascending=False).head(3).index.tolist()
    return stats | {"Más volátiles": ", ".join(vol)}

def build_context_profile(df: pd.DataFrame):
    num = df.select_dtypes(include=np.number)
    if num.empty:
        return "Sin métricas numéricas."
    desc = num.describe().T[["mean","std","min","max"]].round(3)
    # texto compacto
    lines = []
    for c, row in desc.iterrows():
        lines.append(f"{c}: mean={row['mean']}, std={row['std']}, min={row['min']}, max={row['max']}")
    return " | ".join(lines[:10])

def system_prompt(df: pd.DataFrame, persona: str) -> str:
    profile = build_context_profile(df)
    return textwrap.dedent(f"""
    Eres {persona}. Respondes claro, breve y accionable. Experto en macroeconomía, finanzas y mercados.
    Usa el perfil del dataset si el usuario pregunta por los datos; de lo contrario, responde con teoría,
    evidencia empírica general y razonamiento económico transparente.

    Perfil del dataset (para contexto):
    {profile}

    Reglas:
    - Si no hay cifras exactas, declara supuestos y rangos razonables.
    - Evita alucinaciones numéricas. Si calculas algo, indícalo.
    - Ofrece pasos o marcos (IS–LM, AS–AD, regla de Taylor, Mundell-Fleming, etc.) cuando ayuden.
    - Cuando te pidan escenarios, brinda impactos de 1° y 2° orden, y riesgos.
    """)

def ensure_groq():
    key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    return key

def llm_call(model_name, temperature, max_tokens, messages):
    llm = ChatGroq(model=model_name, temperature=temperature, max_tokens=max_tokens)
    resp = llm.invoke(messages)
    return resp.content if hasattr(resp, "content") else str(resp)

def export_chat(history, kind="json"):
    if kind == "json":
        return json.dumps(history, ensure_ascii=False, indent=2).encode("utf-8"), "chat_history.json"
    md = []
    for m in history:
        md.append(("**Tú:** " if m["role"]=="user" else "**Agente:** ") + m["content"])
    return "\n\n".join(md).encode("utf-8"), "chat_history.md"

# ----------------------------- SIDEBAR -----------------------------
with st.sidebar:
    st.markdown("## ⚙️ Ajustes")
    st.markdown("Configura el modelo y el contexto.")

    groq_key = ensure_groq()
    groq_key = st.text_input("GROQ_API_KEY", value=groq_key, type="password",
                             help="Define en st.secrets o variable de entorno.", placeholder="sk_...")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    model = st.selectbox("Modelo Groq", [
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
    ], index=0)
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("Máx. tokens respuesta", 256, 4096, 1024, 64)

    st.divider()
    st.markdown("## 📄 Datos (opcional para contexto del agente)")
    uploaded = st.file_uploader("Cargar CSV (opcional)", type=["csv"])
    rows = st.slider("Filas sintéticas", 90, 730, 365, 30)
    start = st.date_input("Fecha inicial sintética", pd.to_datetime("2024-01-01"))
    seed = st.number_input("Seed", 0, 10_000, 42, 1)

    st.divider()
    st.caption("Hecho con ❤️ • Streamlit + LangChain + Groq ")

# ----------------------------- DATA (solo para perfil de contexto) -----------------------------
df = load_data(uploaded) if uploaded else generate_synthetic_data(num_days=rows, start_date=str(start), seed=seed)
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# ----------------------------- HEADER -----------------------------
st.markdown("💼 EcoAgent Pro ")
kpi = quick_stats(df)
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Filas", kpi.get("Filas", "-"))
with c2: st.metric("Cols num", kpi.get("Cols num", "-"))
with c3: st.metric("Nulos (%)", kpi.get("Nulos (%)", "-"))
with c4: st.metric("Más volátiles", kpi.get("Más volátiles", "-"))

st.write("")

# ----------------------------- TABS -----------------------------
tab_chat, tab_tools, tab_settings = st.tabs(["💬 Chat Económico", "🛠️ Simuladores", "🔧 Ajustes"])

# ====== TAB CHAT ======
with tab_chat:
    if "history" not in st.session_state:
        st.session_state["history"] = []

    persona = st.selectbox("Rol del agente", [
        "un economista senior orientado a política macro",
        "un analista de mercados con foco cuantitativo",
        "un profesor de economía con ejemplos simples"
    ], index=0)

    # Mensajes (burbuja)
    if st.session_state["history"]:
        for msg in st.session_state["history"]:
            role = "Tú" if msg["role"] == "user" else "Agente"
            bubble_class = "user" if msg["role"] == "user" else "ai"
            st.markdown(
                f"""
                <div class="chat">
                  <div class="bubble {bubble_class}">
                    <div class="role">{role}</div>
                    <div>{msg['content']}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with st.form("chat_form", clear_on_submit=True):
        question = st.text_area(
            "Pregunta sobre economía (teoría, mercados, política monetaria/fiscal, comercio, finanzas corporativas, etc.)",
            height=120,
            placeholder="Ej: ¿Qué efectos de 1° y 2° orden tendría subir la tasa en 100 pb en una economía abierta con tipo de cambio flexible?"
        )
        col_s1, col_s2, col_s3 = st.columns([1,1,2])
        with col_s1:
            ask_btn = st.form_submit_button("Enviar 💬", use_container_width=True)
        with col_s2:
            clear_btn = st.form_submit_button("Limpiar 🧹", use_container_width=True)
        with col_s3:
            exp_kind = st.selectbox("Exportar chat", ["—", "JSON", "Markdown"])

    if clear_btn:
        st.session_state["history"] = []
        st.experimental_rerun()

    # Export
    if exp_kind and exp_kind != "—":
        blob, fname = export_chat(st.session_state["history"], "json" if exp_kind == "JSON" else "md")
        st.download_button("Descargar historial", blob, file_name=fname,
                           mime="application/json" if exp_kind=="JSON" else "text/markdown")

    # LLM
    if ask_btn and question:
        if not os.getenv("GROQ_API_KEY"):
            st.error("Falta GROQ_API_KEY. Configúrala en el sidebar o en st.secrets.")
        else:
            sys = system_prompt(df, persona)
            messages = [SystemMessage(content=sys)]
            for h in st.session_state["history"]:
                if h["role"] == "user":
                    messages.append(HumanMessage(content=h["content"]))
                else:
                    messages.append(AIMessage(content=h["content"]))
            messages.append(HumanMessage(content=question))

            try:
                with st.spinner("Pensando..."):
                    answer = llm_call(model, temperature, max_tokens, messages)
            except Exception as exc:
                answer = f"Error del modelo: {exc}"

            st.session_state["history"].append({"role": "user", "content": question})
            st.session_state["history"].append({"role": "ai", "content": answer})

            st.markdown(
                f"""
                <div class="chat">
                  <div class="bubble user">
                    <div class="role">Tú</div>
                    <div>{question}</div>
                  </div>
                </div>
                <div class="chat">
                  <div class="bubble ai">
                    <div class="role">Agente</div>
                    <div>{answer}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ====== TAB TOOLS (Simuladores ligeros, locales) ======
with tab_tools:
    st.markdown("### 🛠️ Simuladores rápidos (offline)")
    sim1, sim2, sim3 = st.columns(3)

    with sim1:
        st.markdown("**📉 Tasa real ex-ante (Fisher simple)**")
        i_nom = st.number_input("Tasa nominal anual (%)", value=6.0, step=0.25, key="i_nom")
        pi_exp = st.number_input("Inflación esperada anual (%)", value=3.0, step=0.25, key="pi_exp")
        real_rate = ((1 + i_nom/100) / (1 + pi_exp/100) - 1) * 100
        st.metric("Tasa real ex-ante", f"{real_rate:.2f}%")

    with sim2:
        st.markdown("**📈 Crecimiento compuesto del PIB**")
        g = st.number_input("Crecimiento anual (%)", value=3.0, step=0.25, key="gdp_g")
        years = st.number_input("Años", value=5, min_value=1, max_value=50, step=1, key="gdp_n")
        base = st.number_input("Nivel base (100=índice)", value=100.0, step=1.0, key="gdp_base")
        future = base * ((1 + g/100) ** years)
        st.metric(f"Nivel en {years} años", f"{future:.2f}")

    with sim3:
        st.markdown("**💸 Dinámica Deuda/PIB (muy simplificada)**")
        debt0 = st.number_input("Deuda/PIB inicial (%)", value=60.0, step=1.0, key="debt0")
        r = st.number_input("Tasa interés efectiva (%)", value=5.0, step=0.25, key="r_eff")
        gdp = st.number_input("Crecimiento PIB (%)", value=3.0, step=0.25, key="g_eff")
        pb = st.number_input("Balance primario (% PIB)", value=-1.0, step=0.25, help=">0 superávit; <0 déficit", key="pb")
        years_d = st.number_input("Años horizonte", value=5, min_value=1, max_value=50, step=1, key="yrs_d")
        # Aproximación: b_{t+1} ≈ b_t * (1 + r - g) + pb
        b = debt0
        for _ in range(int(years_d)):
            b = b * (1 + (r - gdp)/100) - pb  # signo: superávit reduce deuda
        st.metric(f"Deuda/PIB en {int(years_d)} años", f"{b:.1f}%")

    st.caption("Nota: modelos simplificados para intuición; no sustituyen análisis técnico completo.")

# ====== TAB AJUSTES ======
with tab_settings:
    st.markdown("### Utilidades")
    st.download_button("Descargar CSV actual", df.to_csv(index=False).encode("utf-8"),
                       "dataset_context.csv", mime="text/csv")
    st.code(
        """# .streamlit/secrets.toml
GROQ_API_KEY = "coloca_tu_key_aquí"
""", language="toml"
    )
    st.markdown("- Recomendación: 70B para razonamiento conceptual; 8B para respuestas rápidas.")
