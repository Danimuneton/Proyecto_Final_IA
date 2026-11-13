"""Dashboard: Agente de calidad del aire del Valle de Aburrá.

Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from aire import config as C  # noqa: E402
from aire import service  # noqa: E402
from aire.data import active_stations, load_hourly, load_meteo  # noqa: E402
from aire.ica import categoria_ica  # noqa: E402

st.set_page_config(page_title="Aire Aburrá - Agente de calidad del aire", page_icon="🌫️",
                   layout="wide")


# ---------------------------------------------------------------------------
# Carga cacheada
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando datos y modelos...")
def boot():
    service.models()
    service.feature_table()
    service.viterbi_paths()
    return True


@st.cache_resource(show_spinner="Cargando índice RAG (la primera vez descarga el modelo de embeddings)...")
def retriever():
    from aire.rag import get_retriever
    return get_retriever()


@st.cache_resource(show_spinner=False)
def get_agent():
    from aire.agent import AirQualityAgent
    return AirQualityAgent()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


try:
    boot()
except service.ToolError as exc:
    st.error(f"{exc}")
    st.stop()

stations = active_stations()
lo, hi = service.data_range()
M = service.matrix()

# ---------------------------------------------------------------------------
# Barra lateral: parámetros
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Parámetros")
municipios = ["Todos"] + sorted(stations["municipio"].unique().tolist())
mun = st.sidebar.selectbox("Municipio", municipios, index=0)
st_f = stations if mun == "Todos" else stations[stations["municipio"] == mun]
labels = {int(r.codigo): f"{r.nombre_corto} · {r.nombre}" for r in st_f.itertuples()}
default_code = 83 if 83 in labels else next(iter(labels))
code = st.sidebar.selectbox("Estación", list(labels.keys()), format_func=lambda c: labels[c],
                            index=list(labels.keys()).index(default_code))
fecha = st.sidebar.date_input("Fecha de consulta (\"hoy\" en la demo)", value=pd.Timestamp("2024-03-15").date(),
                              min_value=lo.date(), max_value=hi.date())
fecha = pd.Timestamp(fecha)
ventana = st.sidebar.select_slider("Ventana de la serie de tiempo", options=[30, 60, 90, 180, 365], value=90,
                                   format_func=lambda d: f"{d} días")

st.sidebar.markdown("**Escenario para mañana**")
modo = st.sidebar.radio("Meteorología de mañana", ["Observada (reanálisis ERA5)", "Personalizada"],
                        label_visibility="collapsed")
meteo = load_meteo().set_index("fecha")
manana = fecha + pd.Timedelta(days=1)
base = meteo.loc[manana] if manana in meteo.index else meteo.iloc[-1]
esc = {}
if modo == "Personalizada":
    esc["lluvia_mm"] = st.sidebar.slider("Lluvia (mm)", 0.0, 40.0, float(base["lluvia_mm"]), 0.5)
    esc["viento_ms"] = st.sidebar.slider("Viento medio (m/s)", 0.0, 4.0, float(base["viento_ms"]), 0.1)
    esc["radiacion_mj"] = st.sidebar.slider("Radiación solar (MJ/m²)", 5.0, 30.0, float(base["radiacion_mj"]), 0.5)
    esc["nubosidad_pct"] = st.sidebar.slider("Nubosidad (%)", 0.0, 100.0, float(base["nubosidad_pct"]), 1.0)
dom_default = bool(manana.day_of_week == 6)
dom = st.sidebar.checkbox("Mañana es domingo o festivo", value=dom_default)
if dom != dom_default or modo == "Personalizada":
    esc["domingo_o_festivo"] = dom

st.sidebar.divider()
from aire.agent import llm_config  # noqa: E402
cfg = llm_config()
st.sidebar.caption(f"LLM: **{cfg['provider']}** · `{cfg['model']}` · "
                   f"llave {'✅ configurada' if cfg['api_key'] else '❌ falta (ver README)'}")
st.sidebar.caption(f"Datos SIATA: {lo.date()} a {hi.date()}")

# ---------------------------------------------------------------------------
# Encabezado e indicadores
# ---------------------------------------------------------------------------
st.title("🌫️ Agente de calidad del aire del Valle de Aburrá")
st.caption("PM2.5 del SIATA · ICA según Resolución 2254 de 2017 · HMM y regresión logística propios · "
           "Agente LLM con RAG. Herramienta académica (EAFIT, SI3003): no es un pronóstico oficial ni consejo médico.")

info = stations.set_index("codigo").loc[code]
try:
    hoy = service.pm25_station(code, fecha)
except service.ToolError as exc:
    hoy = {"error": str(exc)}
try:
    pred = service.predict_tomorrow(str(code), str(fecha.date()), esc.get("lluvia_mm"), esc.get("viento_ms"),
                                    esc.get("radiacion_mj"), esc.get("nubosidad_pct"),
                                    domingo_o_festivo=esc.get("domingo_o_festivo"))
except service.ToolError as exc:
    pred = {"error": str(exc)}

c1, c2, c3, c4 = st.columns(4)
if "error" not in hoy:
    ica = categoria_ica(hoy["pm25_promedio_24h_ugm3"])
    c1.metric(f"PM2.5 24 h · {info.nombre_corto}", f"{hoy['pm25_promedio_24h_ugm3']} µg/m³",
              help=f"Fecha del dato: {hoy['fecha_dato']}. Norma 24 h: 37 µg/m³; guía OMS: 15 µg/m³.")
    c2.markdown(f"**ICA {ica['ica']}**<br><span style='background:{ica['color']};padding:4px 10px;"
                f"border-radius:6px;color:{'black' if ica['color'] in ('#FFFF00', '#00E400') else 'white'}'>"
                f"{ica['categoria']}</span>", unsafe_allow_html=True)
else:
    c1.warning(hoy["error"])
if "error" not in pred:
    est = max(pred["estado_hmm_hoy"], key=pred["estado_hmm_hoy"].get)
    c3.metric("Estado latente (HMM)", est, help=json.dumps(pred["estado_hmm_hoy"], ensure_ascii=False))
    c4.metric(f"Pronóstico {pred['fecha_pronostico']}", pred["categoria_predicha"].split(" (")[0],
              help=json.dumps(pred["probabilidades_manana"], ensure_ascii=False))
else:
    c3.warning(pred["error"])

tab_map, tab_ts, tab_pred, tab_eval, tab_chat = st.tabs(
    ["🗺️ Mapa", "📈 Serie de tiempo", "🔮 Predicción", "🧪 Modelo y evaluación", "💬 Chat con el agente"])

# ---------------------------------------------------------------------------
# Mapa
# ---------------------------------------------------------------------------
with tab_map:
    vals = M.loc[fecha] if fecha in M.index else pd.Series(dtype=float)
    mp = stations.copy()
    mp["pm25"] = mp["codigo"].map(vals)
    mp["categoria"] = mp["pm25"].map(lambda v: categoria_ica(v)["categoria"])
    mp["texto_pm25"] = mp["pm25"].map(lambda v: "sin dato" if pd.isna(v) else f"{v:.1f} µg/m³")
    mp["tam"] = mp["pm25"].fillna(5).clip(lower=5)
    color_map = {c[4]: c[5] for c in C.ICA_PM25} | {"Sin dato": "#BBBBBB"}
    fig = px.scatter_map(mp, lat="latitud", lon="longitud", color="categoria", size="tam",
                         color_discrete_map=color_map, hover_name="nombre",
                         hover_data={"nombre_corto": True, "municipio": True, "texto_pm25": True,
                                     "tam": False, "latitud": False, "longitud": False},
                         zoom=10.2, center={"lat": 6.25, "lon": -75.57}, height=560,
                         map_style="open-street-map",
                         category_orders={"categoria": list(color_map.keys())})
    sel = mp[mp["codigo"] == code]
    fig.add_trace(go.Scattermap(lat=sel["latitud"], lon=sel["longitud"], mode="markers",
                                marker=dict(size=28, color="rgba(0,0,0,0)"), name="Seleccionada",
                                hoverinfo="skip"))
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), legend_title_text="ICA PM2.5",
                      title=f"PM2.5 promedio 24 h por estación · {fecha.date()}")
    cm, ct = st.columns([3, 2])
    cm.plotly_chart(fig, width="stretch")
    rank = mp.dropna(subset=["pm25"]).sort_values("pm25", ascending=False)[
        ["nombre_corto", "municipio", "pm25", "categoria"]].rename(
        columns={"nombre_corto": "Estación", "municipio": "Municipio", "pm25": "PM2.5 (µg/m³)",
                 "categoria": "Categoría ICA"})
    ct.markdown(f"**Ranking del día** · {len(rank)} estaciones con dato")
    ct.dataframe(rank.style.format({"PM2.5 (µg/m³)": "{:.1f}"}), hide_index=True, width="stretch",
                 height=500)

# ---------------------------------------------------------------------------
# Serie de tiempo
# ---------------------------------------------------------------------------
with tab_ts:
    start = max(lo, fecha - pd.Timedelta(days=ventana - 1))
    s = M[code][(M.index >= start) & (M.index <= fecha)]
    valle = M.mean(axis=1)[(M.index >= start) & (M.index <= fecha)]
    vit = service.viterbi_paths()[code]
    filt = service.filtered_probs()[code]
    idx = [M.index.get_loc(d) for d in s.index]
    fig = go.Figure()
    for (y0, y1, col, name) in [(0, 12.5, "rgba(0,228,0,0.10)", "Buena"), (12.5, 37.5, "rgba(255,255,0,0.12)", "Aceptable"),
                                (37.5, 55.5, "rgba(255,126,0,0.14)", "Dañina GS"), (55.5, 80, "rgba(255,0,0,0.12)", "Dañina")]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=col, line_width=0, annotation_text=name,
                      annotation_position="left", annotation_font_size=10)
    fig.add_trace(go.Scatter(x=valle.index, y=valle, name="Promedio del valle", line=dict(color="gray", dash="dot")))
    fig.add_trace(go.Scatter(x=s.index, y=s, name=f"{info.nombre_corto}", mode="lines+markers",
                             line=dict(color="#1f3b73"),
                             marker=dict(size=7, color=[C.COLORES_ESTADOS[int(np.argmax(filt[i]))] for i in idx]),
                             customdata=[C.ESTADOS_HMM[int(np.argmax(filt[i]))] for i in idx],
                             hovertemplate="%{x|%Y-%m-%d}: %{y:.1f} µg/m³<br>Estado HMM (filtrado): %{customdata}<extra></extra>"))
    fig.add_hline(y=C.NORMA_24H, line_dash="dash", line_color="red", annotation_text="Norma 24 h (37)")
    fig.add_hline(y=C.OMS_24H, line_dash="dash", line_color="green", annotation_text="Guía OMS (15)")
    fig.update_layout(height=430, title=f"PM2.5 diario · {info.nombre} (color del punto = estado HMM)",
                      yaxis_title="µg/m³", legend=dict(orientation="h", y=-0.15), margin=dict(t=50))
    st.plotly_chart(fig, width="stretch")
    st.caption("Puntos: azul = Normal, naranja = Alerta, rojo = Episodio crítico (estado más probable del HMM "
               "filtrado, usando solo datos hasta cada día). Bandas: categorías ICA de la Res. 2254/2017.")

    col_a, col_b = st.columns(2)
    # Probabilidades del HMM en el tiempo
    pf = pd.DataFrame(filt[idx], index=s.index, columns=C.ESTADOS_HMM)
    fig2 = go.Figure()
    for k, name in enumerate(C.ESTADOS_HMM):
        fig2.add_trace(go.Scatter(x=pf.index, y=pf[name], stackgroup="one", name=name,
                                  line=dict(width=0.5, color=C.COLORES_ESTADOS[k])))
    fig2.update_layout(height=300, title="Probabilidad filtrada de cada estado latente (HMM)",
                       yaxis=dict(range=[0, 1]), margin=dict(t=40), legend=dict(orientation="h", y=-0.25))
    col_a.plotly_chart(fig2, width="stretch")
    # Perfil horario del día
    h = load_hourly()
    hh = h[(h["codigo"] == code) & (h["fecha"].dt.normalize() == fecha)]
    fig3 = px.bar(hh, x=hh["fecha"].dt.hour, y="pm25", labels={"x": "Hora", "pm25": "PM2.5 (µg/m³)"},
                  title=f"Perfil horario · {fecha.date()}", height=300)
    fig3.add_hline(y=C.NORMA_24H, line_dash="dash", line_color="red")
    fig3.update_layout(margin=dict(t=40))
    col_b.plotly_chart(fig3, width="stretch")

# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
with tab_pred:
    if "error" in pred:
        st.warning(pred["error"])
    else:
        a, b = st.columns([3, 2])
        pr = pd.DataFrame({"Categoría": list(pred["probabilidades_manana"].keys()),
                           "Probabilidad": list(pred["probabilidades_manana"].values())})
        fig = px.bar(pr, x="Categoría", y="Probabilidad", color="Categoría",
                     color_discrete_sequence=C.COLORES_CLASES, range_y=[0, 1], text_auto=".0%",
                     title=f"Probabilidad de la categoría ICA el {pred['fecha_pronostico']} · {info.nombre_corto}")
        fig.update_layout(showlegend=False, height=380)
        a.plotly_chart(fig, width="stretch")
        b.markdown(f"**Predicción del modelo:** {pred['categoria_predicha']}")
        b.markdown(f"**Línea base (persistencia):** {pred['linea_base_persistencia']}")
        if "valor_real_observado_manana" in pred:
            rv = pred["valor_real_observado_manana"]
            b.markdown(f"**Lo que realmente ocurrió:** {rv['pm25']} µg/m³ → {rv['categoria']}")
        b.markdown("**Escenario usado:** " + ", ".join(f"{k} = {v}" for k, v in pred["escenario_meteorologico_manana"].items()))
        b.markdown("**Factores principales:**")
        for f in pred["factores_principales"]:
            b.markdown(f"- {f['descripcion']}: {f['efecto']} ({f['peso']:+.2f})")
        b.caption(pred["desempeno_historico"])
        b.caption("La regla de decisión corrige el desbalance de clases (ver pestaña de evaluación); por eso la "
                  "categoría predicha puede no ser la de mayor probabilidad.")

        bt_path = C.REPORTS_DIR / "backtest_predicciones.csv"
        if bt_path.exists():
            bt = pd.read_csv(bt_path, parse_dates=["fecha"])
            bts = bt[bt["codigo"] == code].copy()
            bts["fecha_obj"] = bts["fecha"] + pd.Timedelta(days=1)
            if not bts.empty:
                figb = go.Figure()
                figb.add_trace(go.Scatter(x=bts["fecha_obj"], y=bts["clase_manana"], mode="lines", name="Real",
                                          line=dict(color="black", shape="hv")))
                figb.add_trace(go.Scatter(x=bts["fecha_obj"], y=bts["pred_logreg_completo"] + 0.06, mode="markers",
                                          name="Modelo", marker=dict(color="#d62728", size=5)))
                figb.add_trace(go.Scatter(x=bts["fecha_obj"], y=bts["pred_persistencia"] - 0.06, mode="markers",
                                          name="Persistencia", marker=dict(color="#7f7f7f", size=5, symbol="x")))
                figb.update_layout(height=330, title=f"Backtesting fuera de muestra (ene–jul 2024) · {info.nombre_corto}",
                                   yaxis=dict(tickvals=[0, 1, 2], ticktext=C.CLASES_CORTAS), margin=dict(t=40),
                                   legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(figb, width="stretch")
                acc_m = (bts["pred_logreg_completo"] == bts["clase_manana"]).mean()
                acc_p = (bts["pred_persistencia"] == bts["clase_manana"]).mean()
                st.caption(f"Exactitud en esta estación: modelo {acc_m:.1%} · persistencia {acc_p:.1%} "
                           f"({len(bts)} días).")

# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
with tab_eval:
    met = load_json(C.REPORTS_DIR / "metricas_parte1.json")
    st.subheader("Parte 1: predicción de la categoría del día siguiente")
    if met:
        rows = [{"Modelo": k, "Exactitud": v["accuracy"], "Exact. balanceada": v["balanced_accuracy"],
                 "F1 macro": v["macro_f1"], "Recall Dañina": v["recall_por_clase"][2],
                 "Log-loss": v.get("log_loss"), "Brier": v.get("brier")} for k, v in met["resultados"].items()]
        st.dataframe(pd.DataFrame(rows).style.format(precision=3, na_rep="—"), hide_index=True,
                     width="stretch")
        st.caption(met["descripcion"] + f" n = {met['n_predicciones']} predicciones estación-día.")
        c1, c2 = st.columns(2)
        cmx = np.array(met["resultados"][list(met["resultados"])[-1]]["matriz_confusion"])
        figc = px.imshow(cmx, x=C.CLASES_CORTAS, y=C.CLASES_CORTAS, text_auto=True, color_continuous_scale="Blues",
                         labels=dict(x="Predicho", y="Real", color="n"), title="Matriz de confusión (modelo final)")
        c1.plotly_chart(figc, width="stretch")
        A = np.array(met["modelo_final"]["hmm"]["A"])
        figa = px.imshow(A, x=C.ESTADOS_HMM, y=C.ESTADOS_HMM, text_auto=".3f", color_continuous_scale="Oranges",
                         labels=dict(x="Estado mañana", y="Estado hoy", color="P"),
                         title="Matriz de transición del HMM (Baum-Welch propio)")
        c2.plotly_chart(figa, width="stretch")
        mu = np.exp(np.array(met["modelo_final"]["hmm"]["mu"]))
        st.caption("Medias de emisión del HMM (µg/m³): " + ", ".join(f"{n} ≈ {m:.1f}" for n, m in zip(C.ESTADOS_HMM, mu)))
        md = C.REPORTS_DIR / "metricas_parte1.md"
        with st.expander("Detalle: log-verosimilitud del HMM, intervalos bootstrap y verificación con scikit-learn"):
            st.markdown(md.read_text(encoding="utf-8") if md.exists() else "—")
    else:
        st.info("Ejecute `python scripts/train_models.py` para generar las métricas.")

    st.subheader("Parte 2: evaluación del RAG y del agente")
    rag = C.REPORTS_DIR / "evaluacion_rag.md"
    if rag.exists():
        st.markdown("**Recuperación (18 consultas con documento esperado):**")
        st.markdown(rag.read_text(encoding="utf-8"))
    ev = C.EVAL_DIR / "resultados_agente.md"
    if ev.exists():
        with st.expander("Resultados de las preguntas de prueba del agente", expanded=False):
            st.markdown(ev.read_text(encoding="utf-8"))
    else:
        st.info("Ejecute `python scripts/evaluate_agent.py` para evaluar el agente (requiere llave del LLM).")

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
with tab_chat:
    st.markdown(f"Pregúntale al agente. Contexto: **{info.nombre_corto}** ({info.municipio}), "
                f"fecha de consulta **{fecha.date()}**.")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    sugeridas = ["¿Cómo está el aire hoy en esta estación y qué significa?",
                 "¿Puedo salir a trotar mañana? Tengo asma.",
                 "¿Está el valle en un episodio de contaminación?",
                 "¿Qué medidas contempla el POECA en un episodio crítico?"]
    cols = st.columns(len(sugeridas))
    clicked = None
    for c, q in zip(cols, sugeridas):
        if c.button(q, width="stretch"):
            clicked = q
    if st.button("🗑️ Borrar conversación"):
        st.session_state.chat = []

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("modelo"):
                st.caption(m["modelo"])
            if m.get("steps"):
                with st.expander(f"🔧 Herramientas usadas ({len(m['steps'])})"):
                    for s in m["steps"]:
                        st.markdown(f"**{s['tool']}** `{json.dumps(s['args'], ensure_ascii=False)}` → "
                                    f"{'✅' if s['ok'] else '⚠️ ' + s.get('error', '')}")
            if m.get("sources"):
                with st.expander("📚 Fuentes consultadas"):
                    for src in m["sources"]:
                        st.markdown(f"- {src['cita']} {src['titulo']} {('— ' + src['url']) if src.get('url') else ''}")

    prompt = st.chat_input("Escribe tu pregunta sobre la calidad del aire...") or clicked
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Consultando herramientas..."):
                try:
                    retriever()
                    agent = get_agent()
                    hist = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat[:-1]]
                    res = agent.run(prompt, history=hist, fecha=str(fecha.date()), estacion=info.nombre_corto)
                    answer, steps, sources = res.answer, res.steps, res.sources
                    modelo = f"Modelo: `{res.model}` · {res.seconds} s · {res.iterations} iteraciones"
                except Exception as exc:  # noqa: BLE001
                    answer, steps, sources, modelo = f"⚠️ {exc}", [], [], ""
            st.markdown(answer)
            if modelo:
                st.caption(modelo)
            steps_d = [{"tool": s.tool, "args": s.args, "ok": s.ok, "error": s.observation.get("error", "")}
                       for s in steps]
            if steps_d:
                with st.expander(f"🔧 Herramientas usadas ({len(steps_d)})"):
                    for s in steps_d:
                        st.markdown(f"**{s['tool']}** `{json.dumps(s['args'], ensure_ascii=False)}` → "
                                    f"{'✅' if s['ok'] else '⚠️ ' + s['error']}")
            if sources:
                with st.expander("📚 Fuentes consultadas"):
                    for src in sources:
                        st.markdown(f"- {src['cita']} {src['titulo']} {('— ' + src['url']) if src.get('url') else ''}")
        st.session_state.chat.append({"role": "assistant", "content": answer, "steps": steps_d, "sources": sources,
                                      "modelo": modelo})
