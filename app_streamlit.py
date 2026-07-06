"""Demo interactiva: cliente visual de la API de inferencia.

Uso (con la API corriendo en :8000):
    uv run streamlit run app_streamlit.py

La demo NO carga el modelo: consume la API (separación producto/demo).
"""

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Text Detector", page_icon="🕵️", layout="centered")
st.title("🕵️ AI Text Detector")
st.caption("¿Este ensayo lo escribió un humano o un LLM? — demo del servicio de inferencia")

with st.sidebar:
    st.subheader("Servicio")
    try:
        health = httpx.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"API OK — modelo: `{health['model_name']}`")
    except Exception:
        st.error(f"API no disponible en {API_URL}. Levántala con `make serve`.")
        st.stop()
    st.markdown(f"[Documentación de la API]({API_URL}/docs)")

text = st.text_area(
    "Pega un ensayo (en inglés, como los de la competencia):",
    height=260,
    placeholder="Phones and driving is a topic that many people debate about...",
)

if st.button("Analizar", type="primary") and text.strip():
    with st.spinner("Consultando al detector..."):
        r = httpx.post(f"{API_URL}/predict", json={"text": text}, timeout=30)
    if r.status_code != 200:
        st.error(f"Error {r.status_code}: {r.text}")
    else:
        res = r.json()
        proba = res["proba_ai"]
        col1, col2 = st.columns(2)
        col1.metric("P(generado por IA)", f"{proba:.1%}")
        col2.metric("Veredicto", "🤖 IA" if res["label"] == 1 else "✍️ Humano")
        st.progress(proba)
        st.caption(
            f"Modelo: `{res['model_name']}` · Latencia: {res['latency_ms']:.0f} ms · "
            "Umbral: 0.5 · Las predicciones se registran (sin el texto) para monitoreo de drift."
        )
