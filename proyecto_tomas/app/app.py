"""Dashboard: calidad del aire (PM2.5) en el Valle de Aburrá + agente LLM con RAG.
Ejecutar:  streamlit run app/app.py
"""
import json
import pathlib
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
import modelo as nb      # noqa: E402
import agente            # noqa: E402

COLORES = {"Buena": "#2e9e4f", "Aceptable": "#e0b400", "Dañina a sensibles o peor": "#e8761c"}

st.set_page_config(page_title="Aire del Valle de Aburrá", page_icon="🌫️", layout="wide")


@st.cache_data
def cargar():
    df = nb.cargar_datos()
    est = pd.read_csv(BASE / "data" / "processed" / "estaciones.csv")
    est = est[est["estacion"].isin(df["estacion"].unique())]
    return df, est


def categoria(v):
    return nb.CLASES[int(nb.categorizar([v])[0])]


df, est = cargar()
f_min, f_max = df["fecha"].min().date(), df["fecha"].max().date()

# ------------------------------------------------------------ barra lateral: parámetros del usuario
st.sidebar.header("Parámetros")
nombres = est.sort_values("nombre")["nombre"].tolist()
nombre_sel = st.sidebar.selectbox("Estación", nombres, index=0)
cod = int(est.loc[est["nombre"] == nombre_sel, "estacion"].iloc[0])
serie = df[df["estacion"] == cod].set_index("fecha")["pm25"]
fecha = pd.Timestamp(st.sidebar.date_input("Fecha", value=serie.index.max().date(),
                                           min_value=serie.index.min().date(), max_value=serie.index.max().date()))
ventana = st.sidebar.select_slider("Ventana de la serie (días)", options=[14, 30, 60, 90, 180, 365], value=60)
st.sidebar.caption(f"Datos horarios de SIATA promediados por día, del {f_min} al {f_max}. "
                   "Es un histórico: 'mañana' significa el día siguiente a la fecha elegida.")

st.title("🌫️ Calidad del aire (PM2.5) en el Valle de Aburrá")
st.caption("Proyecto final SI3003 – Inteligencia Artificial, EAFIT. Información orientativa; no es consejo médico, legal ni de seguridad.")

tab_serie, tab_mapa, tab_pred, tab_chat = st.tabs(["📈 Serie de tiempo", "🗺️ Mapa y comparación", "🔮 Predicción de mañana", "💬 Agente"])

# ------------------------------------------------------------ 1) serie de tiempo
with tab_serie:
    if fecha not in serie.index:
        st.warning(f"No hay dato de {nombre_sel} el {fecha.date()}. Elige otra fecha.")
    else:
        v = float(serie[fecha])
        prom7 = float(serie[(serie.index <= fecha) & (serie.index > fecha - pd.Timedelta(days=7))].mean())
        c1, c2, c3 = st.columns(3)
        c1.metric(f"PM2.5 el {fecha.date()}", f"{v:.1f} µg/m³")
        c2.metric("Categoría ICA", categoria(v))
        c3.metric("Promedio de 7 días", f"{prom7:.1f} µg/m³")
    ini = fecha - pd.Timedelta(days=ventana)
    s = serie[(serie.index >= ini) & (serie.index <= fecha)]
    fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines", name="PM2.5 diario", line=dict(width=2)))
    fig.add_hline(y=nb.CORTES[0], line_dash="dash", line_color=COLORES["Buena"], annotation_text="12 (límite Buena)")
    fig.add_hline(y=nb.CORTES[1], line_dash="dash", line_color=COLORES["Dañina a sensibles o peor"], annotation_text="37 (límite Aceptable)")
    if fecha in s.index:
        fig.add_trace(go.Scatter(x=[fecha], y=[s[fecha]], mode="markers", name="Fecha elegida", marker=dict(size=12, color="red")))
    fig.update_layout(title=f"{nombre_sel}", xaxis_title="Fecha", yaxis_title="PM2.5 (µg/m³, promedio diario)",
                      height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch", key="grafico_serie")
    st.caption("Líneas punteadas: puntos de corte de PM2.5 (24 h) del ICA, Resolución 2254 de 2017.")

# ------------------------------------------------------------ 2) mapa y comparación entre estaciones
with tab_mapa:
    dia = df[df["fecha"] == fecha].merge(est[["estacion", "lat", "lon"]], on="estacion")
    if dia.empty:
        st.warning(f"No hay datos de ninguna estación el {fecha.date()}.")
    else:
        dia["categoria"] = dia["pm25"].map(categoria)
        dia["tamano"] = dia["pm25"].clip(lower=3)
        mapa_fn = getattr(px, "scatter_map", None) or px.scatter_mapbox
        fig_m = mapa_fn(dia, lat="lat", lon="lon", color="categoria", size="tamano", size_max=22, zoom=9.8,
                        center={"lat": float(dia["lat"].mean()), "lon": float(dia["lon"].mean())},
                        hover_name="nombre", hover_data={"pm25": ":.1f", "lat": False, "lon": False, "tamano": False, "categoria": False},
                        color_discrete_map=COLORES, height=560)
        fig_m.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=30, b=0), title=f"Estaciones el {fecha.date()}") \
            if hasattr(px, "scatter_map") else fig_m.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_m, width="stretch", key="grafico_mapa")
        orden = dia.sort_values("pm25")
        fig_b = px.bar(orden, x="pm25", y="nombre", color="categoria", orientation="h", color_discrete_map=COLORES,
                       labels={"pm25": "PM2.5 (µg/m³)", "nombre": ""}, height=max(380, 26 * len(orden)),
                       title="Comparación entre estaciones")
        fig_b.add_vline(x=nb.CORTES[0], line_dash="dash")
        fig_b.add_vline(x=nb.CORTES[1], line_dash="dash")
        st.plotly_chart(fig_b, width="stretch", key="grafico_comparacion")

# ------------------------------------------------------------ 3) predicción
with tab_pred:
    st.subheader(f"{nombre_sel}")
    r = nb.predecir_manana(cod, fecha)
    if "error" in r:
        st.warning(r["error"] + " (hacen falta datos de la fecha elegida y del día anterior)")
    else:
        c1, c2 = st.columns(2)
        c1.metric(f"Categoría predicha para el {r['fecha_prediccion']}", r["categoria_predicha"])
        c2.metric(f"Categoría observada el {r['fecha_base']}", r["categoria_base"], f"{r['pm25_base']} µg/m³", delta_color="off")
        pr = pd.DataFrame({"categoria": list(r["probabilidades"]), "probabilidad": list(r["probabilidades"].values())})
        fig_p = px.bar(pr, x="categoria", y="probabilidad", color="categoria", color_discrete_map=COLORES, range_y=[0, 1], height=320)
        fig_p.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_p, width="stretch", key="grafico_prediccion")
        real = serie.get(fecha + pd.Timedelta(days=1))
        if real is not None:
            st.info(f"Valor observado al día siguiente: {real:.1f} µg/m³ → {categoria(real)}.")
    mp = BASE / "models" / "metricas.json"
    if mp.exists():
        m = json.load(open(mp))
        st.markdown(f"**Desempeño del modelo (periodo de prueba, desde {m['corte_prueba']}):**")
        tabla = pd.DataFrame([{"Modelo": k, "Accuracy": round(v["accuracy"], 3), "F1 macro": round(v["f1_macro"], 3)}
                              for k, v in m["prueba"].items()])
        st.dataframe(tabla, hide_index=True, width="stretch")
        st.caption("Naive Bayes escrito a mano frente a líneas base. En el periodo de prueba no hubo días 'Dañina+', "
                   "así que esa clase no se puede evaluar. Ver el informe para las limitaciones.")

# ------------------------------------------------------------ 4) chat con el agente
with tab_chat:
    st.caption(f"Contexto enviado al agente: estación **{nombre_sel}**, fecha **{fecha.date()}** (\"hoy\" en el histórico).")
    st.caption("Pregunta por valores medidos, predicciones o el significado de las categorías. Ejemplos: "
               "«¿Cómo estará mañana el aire en Bello?», «¿Qué es el POECA?», «¿Qué significa un ICA naranja?»")
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []
    for m in st.session_state.mensajes:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("trazas"):
                with st.expander("Herramientas usadas"):
                    for t in m["trazas"]:
                        st.code(f"{t['herramienta']}({json.dumps(t['argumentos'], ensure_ascii=False)})")
    pregunta = st.chat_input("Escribe tu pregunta")
    if pregunta:
        historial = [{"role": m["role"], "content": m["content"]} for m in st.session_state.mensajes][-6:]
        st.session_state.mensajes.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    texto, trazas = agente.responder(
                        pregunta, historial, contexto={"estacion": nombre_sel, "fecha": str(fecha.date())})
                except Exception as e:
                    texto, trazas = f"No pude contactar al modelo de lenguaje: {e}", []
            st.markdown(texto)
            if trazas:
                with st.expander("Herramientas usadas"):
                    for t in trazas:
                        st.code(f"{t['herramienta']}({json.dumps(t['argumentos'], ensure_ascii=False)})")
        st.session_state.mensajes.append({"role": "assistant", "content": texto, "trazas": trazas})