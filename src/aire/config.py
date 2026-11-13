"""Configuración central: rutas, estaciones, categorías normativas y parámetros."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
CORPUS_DIR = ROOT / "docs" / "corpus"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EVAL_DIR = ROOT / "evaluacion"

# ---------------------------------------------------------------------------
# Fuentes de datos (ver README, sección "Datos")
# ---------------------------------------------------------------------------
SIATA_PM25_URL = "https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_pm25.json"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
# Punto representativo del centro del valle (Medellín, La Alpujarra aprox.)
METEO_LAT, METEO_LON = 6.25, -75.57

PM25_HOURLY_FILE = PROCESSED_DIR / "pm25_horario.csv.gz"
PM25_DAILY_FILE = PROCESSED_DIR / "pm25_diario.csv"
STATIONS_FILE = PROCESSED_DIR / "estaciones.csv"
METEO_DAILY_FILE = PROCESSED_DIR / "meteo_diaria.csv"

# Una hora es válida si SIATA le asigna bandera de calidad <= 2.5 y valor >= 0
# (criterio publicado por SIATA para datos de buena calidad).
MAX_QUALITY_FLAG = 2.5
# Un promedio de 24 h es representativo si tiene al menos 75 % de datos horarios
# (18 de 24), criterio usual del Protocolo de Monitoreo de Calidad del Aire.
MIN_HOURS_PER_DAY = 18

# Municipio de cada estación (por código serial SIATA)
MUNICIPIO_POR_CODIGO = {
    3: "Girardota", 12: "Medellín", 25: "Medellín", 28: "Itagüí", 31: "Caldas",
    38: "Itagüí", 44: "Medellín", 48: "Medellín", 69: "Caldas", 78: "La Estrella",
    79: "Medellín", 80: "Medellín", 81: "Barbosa", 82: "Copacabana", 83: "Medellín",
    84: "Medellín", 85: "Medellín", 86: "Medellín", 87: "Bello", 88: "Envigado",
    90: "Sabaneta",
}

# ---------------------------------------------------------------------------
# Índice de Calidad del Aire (ICA) para PM2.5, 24 h. Resolución 2254 de 2017,
# Tabla 6 (puntos de corte) y Tabla 5 (colores y estados).
# (conc_baja, conc_alta, ica_bajo, ica_alto, categoría, color hex)
# ---------------------------------------------------------------------------
ICA_PM25 = [
    (0, 12, 0, 50, "Buena", "#00E400"),
    (13, 37, 51, 100, "Aceptable", "#FFFF00"),
    (38, 55, 101, 150, "Dañina a la salud de grupos sensibles", "#FF7E00"),
    (56, 150, 151, 200, "Dañina a la salud", "#FF0000"),
    (151, 250, 201, 300, "Muy dañina a la salud", "#8F3F97"),
    (251, 500, 301, 500, "Peligrosa", "#7E0023"),
]

# Clases que predice el clasificador del día siguiente (agrupación de la Tabla 6).
# La clase 2 coincide con el umbral de "Prevención" de la Tabla 4 (>= 38 ug/m3).
CLASES = ["Buena", "Aceptable", "Dañina (grupos sensibles o peor)"]
CLASES_CORTAS = ["Buena", "Aceptable", "Dañina"]
COLORES_CLASES = ["#00E400", "#E6C700", "#FF7E00"]

# Estados latentes del HMM, ordenados por media creciente de log(PM2.5)
ESTADOS_HMM = ["Normal", "Alerta", "Episodio crítico"]
COLORES_ESTADOS = ["#2E86AB", "#F4A261", "#C0392B"]

# Norma colombiana (Res. 2254/2017) y guía OMS 2021 para PM2.5
NORMA_24H = 37.0
NORMA_ANUAL = 25.0
OMS_24H = 15.0
OMS_ANUAL = 5.0

# ---------------------------------------------------------------------------
# Agente
# ---------------------------------------------------------------------------
T_MAX = 6  # número máximo de iteraciones del loop agéntico
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RAG_INDEX_DIR = MODELS_DIR / "rag_index"
