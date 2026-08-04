# -*- coding: utf-8 -*-
"""
Geoportal Streamlit - SAT de Sequía Agrícola, Microcuenca Amatitán.
Versión final para GitHub + Streamlit Cloud, sin geemap.

Motor:
- Google Earth Engine
- CHIRPS para precipitación y SPI-3
- MODIS MOD13Q1 para VCI
- FABDEM para pendiente
- IISS y alerta integrada
"""

import datetime
import json

import ee
import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st_stats
import streamlit as st
from streamlit_folium import st_folium


# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================

APP_TITLE = "SAT de Sequía Agrícola - Microcuenca Amatitán"
PROJECT_ID_DEFAULT = "micuencaamatitan"

RUTA_CUENCA = "projects/micuencaamatitan/assets/MicrocuencaAmatitan"
RUTA_DRENAJE = "projects/micuencaamatitan/assets/RiosMicrocuencaAmatitan"
RUTA_DEM = "projects/micuencaamatitan/assets/FABDEM_TITIHUAPA"

CRS_METRICO = "EPSG:32616"
ESCALA_DEM = 30
ESCALA_CHIRPS = 5566
ESCALA_MODIS = 250

ANIO_BASE_SPI_INICIO = 1981
ANIO_BASE_SPI_FIN = 2026
ANIO_BASE_MODIS_INICIO = 2000
ANIO_BASE_MODIS_FIN = 2026

NOMBRES_ALERTA = {
    0: "Normal",
    1: "Vigilancia",
    2: "Prealerta",
    3: "Alerta",
    4: "Emergencia",
}

COLORES_ALERTA = {
    0: "#1a9850",
    1: "#91cf60",
    2: "#fee08b",
    3: "#fc8d59",
    4: "#b2182b",
}

UMBRALES_ESTANDAR = {
    "SPI3": {
        "vigilancia": -0.5,
        "prealerta": -1.0,
        "alerta": -1.5,
        "emergencia": -2.0,
    },
    "VCI": {
        "vigilancia": 50,
        "prealerta": 40,
        "alerta": 30,
        "emergencia": 20,
    },
}

UMBRALES_SENSIBLES = {
    "SPI3": {
        "vigilancia": -0.2,
        "prealerta": -0.7,
        "alerta": -1.2,
        "emergencia": -1.8,
    },
    "VCI": {
        "vigilancia": 60,
        "prealerta": 45,
        "alerta": 35,
        "emergencia": 25,
    },
}


# =============================================================================
# GOOGLE EARTH ENGINE
# =============================================================================

@st.cache_resource(show_spinner=False)
def inicializar_gee(project_id: str):
    """
    Inicializa Google Earth Engine.

    Modo local:
        earthengine authenticate
        streamlit run app.py

    Modo Streamlit Cloud:
        Configurar secrets con cuenta de servicio:
        [gee]
        service_account = "..."
        project = "micuencaamatitan"
        private_key = "TU_LLAVE_PRIVADA"
    """
    try:
        if "gee" in st.secrets:
            service_account = st.secrets["gee"]["service_account"]
            private_key = st.secrets["gee"]["private_key"]
            project = st.secrets["gee"].get("project", project_id)
            credentials = ee.ServiceAccountCredentials(service_account, key_data=private_key)
            ee.Initialize(credentials, project=project)
        elif "EARTHENGINE_CREDENTIALS" in st.secrets:
            creds = json.loads(st.secrets["EARTHENGINE_CREDENTIALS"])
            credentials = ee.ServiceAccountCredentials(
                creds["client_email"],
                key_data=json.dumps(creds),
            )
            ee.Initialize(credentials, project=project_id)
        else:
            ee.Initialize(project=project_id)

        return True, "Google Earth Engine conectado."
    except Exception as exc:
        return False, f"No se pudo conectar a Google Earth Engine: {exc}"


@st.cache_resource(show_spinner=False)
def cargar_assets():
    """Carga microcuenca, red de drenaje y DEM desde Earth Engine Assets."""
    microcuenca = ee.FeatureCollection(RUTA_CUENCA)
    geom = microcuenca.geometry()
    drenaje = ee.FeatureCollection(RUTA_DRENAJE).filterBounds(geom)

    dem_raw = ee.Image(RUTA_DEM)
    dem_band = dem_raw.bandNames().get(0)
    dem = dem_raw.select([dem_band]).rename("elevation").clip(geom)

    return microcuenca, geom, drenaje, dem


# =============================================================================
# INDICADORES DEL SAT
# =============================================================================

def calcular_pendiente(dem, geom):
    """Calcula pendiente en grados usando proyección métrica UTM 16N."""
    dem_metric = dem.reproject(crs=CRS_METRICO, scale=ESCALA_DEM)
    return ee.Terrain.slope(dem_metric).rename("Slope").clip(geom)


def lluvia_mensual_feature(year: int, month: int, geom):
    """Devuelve un ee.Feature con lluvia mensual promedio CHIRPS."""
    inicio = ee.Date.fromYMD(year, month, 1)
    fin = inicio.advance(1, "month")
    lluvia = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
              .filterDate(inicio, fin)
              .select("precipitation")
              .sum()
              .clip(geom)
             )
    
    # reduceRegion devuelve {} si la imagen no tiene bandas (mes sin datos)
    dict_media = lluvia.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=ESCALA_CHIRPS,
        maxPixels=1e9,
    )
    
    # Combinamos con un diccionario por defecto para evitar el error de clave inexistente
    dict_seguro = ee.Dictionary({"precipitation": -9999}).combine(dict_media)
    lluvia_media = dict_seguro.get("precipitation")
    
    return ee.Feature(None, {
        "date": inicio.format("YYYY-MM-dd"),
        "year": year,
        "month": month,
        "rainfall": lluvia_media
    })

@st.cache_data(show_spinner=False)
def construir_serie_chirps(project_id: str, anio_inicio: int, anio_fin: int):
    # ... (Mantén tu código inicial de la función igual) ...
    
    features = []
    for year in range(anio_inicio, anio_fin + 1):
        for month in range(1, 13):
            features.append(lluvia_mensual_feature(year, month, geom))
            
    datos = ee.FeatureCollection(features).getInfo().get("features", [])
    df = pd.DataFrame([f["properties"] for f in datos])
    
    if df.empty:
        return pd.DataFrame(columns=["date", "year", "month", "rainfall"])
        
    df["date"] = pd.to_datetime(df["date"])
    df["rainfall"] = pd.to_numeric(df["rainfall"], errors="coerce")
    
    # NUEVA LÍNEA: Convertimos el flag de no-data a un nulo real de Pandas
    df["rainfall"] = df["rainfall"].replace(-9999, np.nan)
    
    df = df.dropna(subset=["rainfall"]).sort_values("date").reset_index(drop=True)
    return df

def calcular_spi3_gamma(df_lluvia: pd.DataFrame):
    """Calcula SPI-3 con ajuste Gamma por mes calendario."""
    df = df_lluvia.copy().sort_values("date").reset_index(drop=True)
    df["rain_3m"] = df["rainfall"].rolling(window=3, min_periods=3).sum()
    df["month_end"] = df["date"].dt.month
    df["SPI3"] = np.nan

    parametros = {}

    for mes in range(1, 13):
        idx = df["month_end"] == mes
        valores = df.loc[idx, "rain_3m"].dropna()
        if len(valores) < 10:
            continue

        prob_cero = (valores <= 0).sum() / len(valores)
        valores_pos = valores[valores > 0]
        if len(valores_pos) < 10:
            continue

        shape, loc, scale = st_stats.gamma.fit(valores_pos, floc=0)
        parametros[mes] = {"shape": shape, "loc": loc, "scale": scale, "prob_cero": prob_cero}

        cdf_gamma = st_stats.gamma.cdf(df.loc[idx, "rain_3m"], shape, loc=loc, scale=scale)
        cdf = prob_cero + (1 - prob_cero) * cdf_gamma
        df.loc[idx, "SPI3"] = st_stats.norm.ppf(np.clip(cdf, 0.0001, 0.9999))

    return df, parametros


def calcular_spi3_actual(geom, parametros):
    """Calcula SPI-3 de los últimos tres meses completos."""
    hoy = datetime.datetime.now()
    fecha_fin = ee.Date.fromYMD(hoy.year, hoy.month, 1)
    fecha_inicio = fecha_fin.advance(-3, "month")
    
    lluvia = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
              .filterDate(fecha_inicio, fecha_fin)
              .select("precipitation")
              .sum()
             )
    
    dict_3m = lluvia.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=ESCALA_CHIRPS,
        maxPixels=1e9,
    )
    
    # Extracción segura
    lluvia_3m = ee.Dictionary({"precipitation": -9999}).combine(dict_3m).get("precipitation").getInfo()
    if lluvia_3m == -9999:
        lluvia_3m = None

    mes_ref = int((datetime.datetime(hoy.year, hoy.month, 1) - pd.DateOffset(months=1)).month)
    fecha_inicio_txt = fecha_inicio.format("YYYY-MM-dd").getInfo()
    fecha_fin_txt = fecha_fin.format("YYYY-MM-dd").getInfo()
    params = parametros.get(mes_ref)

    if params is None or lluvia_3m is None:
        return None, lluvia_3m, mes_ref, fecha_inicio_txt, fecha_fin_txt

    cdf_gamma = st_stats.gamma.cdf(lluvia_3m, params["shape"], loc=params["loc"], scale=params["scale"])
    cdf = params["prob_cero"] + (1 - params["prob_cero"]) * cdf_gamma
    spi3 = st_stats.norm.ppf(np.clip(cdf, 0.0001, 0.9999))
    return spi3, lluvia_3m, mes_ref, fecha_inicio_txt, fecha_fin_txt


def clasificar_spi3(spi3: float, umbrales: dict):
    """Clasifica SPI-3 en nivel de alerta."""
    if spi3 is None:
        return 0, "Sin datos SPI-3"
    if spi3 <= umbrales["SPI3"]["emergencia"]:
        return 4, "Emergencia climática"
    if spi3 <= umbrales["SPI3"]["alerta"]:
        return 3, "Alerta climática"
    if spi3 <= umbrales["SPI3"]["prealerta"]:
        return 2, "Prealerta climática"
    if spi3 <= umbrales["SPI3"]["vigilancia"]:
        return 1, "Vigilancia climática"
    return 0, "Condición climática normal"


def obtener_mes_modis_disponible(geom, max_retroceso: int = 6):
    """Busca el último mes cerrado con MODIS disponible."""
    hoy = datetime.datetime.now()
    for i in range(1, max_retroceso + 1):
        fecha = datetime.datetime(hoy.year, hoy.month, 1) - pd.DateOffset(months=i)
        year = int(fecha.year)
        month = int(fecha.month)
        inicio = ee.Date.fromYMD(year, month, 1)
        fin = inicio.advance(1, "month")
        col = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(inicio, fin).filterBounds(geom).select("NDVI")
        if col.size().getInfo() > 0:
            return year, month
    return None, None


def calcular_vci_mes(geom, year: int, month: int):
    """Calcula VCI mensual usando MODIS NDVI."""
    if year is None or month is None:
        return None

    inicio = ee.Date.fromYMD(year, month, 1)
    fin = inicio.advance(1, "month")
    col_actual = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(inicio, fin).filterBounds(geom).select("NDVI")

    if col_actual.size().getInfo() == 0:
        return None

    ndvi_actual = col_actual.median().multiply(0.0001).rename("NDVI_actual").clip(geom)

    hist = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterBounds(geom)
        .filter(ee.Filter.calendarRange(ANIO_BASE_MODIS_INICIO, ANIO_BASE_MODIS_FIN, "year"))
        .filter(ee.Filter.calendarRange(month, month, "month"))
        .select("NDVI")
        .map(lambda img: img.multiply(0.0001).copyProperties(img, ["system:time_start"]))
    )

    if hist.size().getInfo() == 0:
        return None

    ndvi_min = hist.min().rename("NDVI_min").clip(geom)
    ndvi_max = hist.max().rename("NDVI_max").clip(geom)
    den = ndvi_max.subtract(ndvi_min).where(ndvi_max.subtract(ndvi_min).eq(0), 0.0001)

    return ndvi_actual.subtract(ndvi_min).divide(den).multiply(100).clamp(0, 100).rename("VCI").clip(geom)


def promedio_imagen(img, geom, banda: str, escala: int):
    """Calcula promedio espacial de una imagen."""
    if img is None:
        return None
    try:
        return img.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=escala, maxPixels=1e9).get(banda).getInfo()
    except Exception:
        return None


def clasificar_vci(vci: float, umbrales: dict):
    """Clasifica VCI en nivel de alerta."""
    if vci is None:
        return 0, "Sin datos VCI"
    if vci < umbrales["VCI"]["emergencia"]:
        return 4, "Emergencia vegetativa"
    if vci < umbrales["VCI"]["alerta"]:
        return 3, "Alerta vegetativa"
    if vci < umbrales["VCI"]["prealerta"]:
        return 2, "Prealerta vegetativa"
    if vci < umbrales["VCI"]["vigilancia"]:
        return 1, "Vigilancia vegetativa"
    return 0, "Condición vegetativa normal"


def clasificar_vci_imagen(vci, geom):
    """Clasifica VCI como imagen categórica conservando proyección."""
    if vci is None:
        # Si no hay datos, creamos una constante pero le forzamos la proyección correcta
        return ee.Image.constant(0).reproject(crs=CRS_METRICO, scale=ESCALA_MODIS).rename("VCI_clase").clip(geom)
    
    # Heredamos la proyección multiplicando por 0
    base = vci.multiply(0)
    return (
        base
        .where(vci.lte(50).And(vci.gt(40)), 1)
        .where(vci.lte(40).And(vci.gt(30)), 2)
        .where(vci.lte(30).And(vci.gt(20)), 3)
        .where(vci.lte(20), 4)
        .rename("VCI_clase")
        .clip(geom)
    )


def normalizar_imagen(img, geom, nombre_banda: str, escala: int):
    """Normaliza imagen entre 0 y 1."""
    stats = img.reduceRegion(reducer=ee.Reducer.minMax(), geometry=geom, scale=escala, maxPixels=1e9)
    min_val = ee.Number(stats.get(f"{nombre_banda}_min"))
    max_val = ee.Number(stats.get(f"{nombre_banda}_max"))
    den = max_val.subtract(min_val).max(0.0001)
    return img.subtract(min_val).divide(den).clamp(0, 1)


def calcular_iiss(geom, pendiente):
    """Calcula IISS con NDVI P10 histórico y pendiente."""
    hist = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterDate(f"{ANIO_BASE_MODIS_INICIO}-01-01", f"{ANIO_BASE_MODIS_FIN}-12-31")
        .filterBounds(geom)
        .select("NDVI")
        .map(lambda img: img.multiply(0.0001).copyProperties(img, ["system:time_start"]))
    )

    ndvi_p10 = hist.reduce(ee.Reducer.percentile([10])).rename("NDVI_P10").clip(geom)
    vuln_veg = ee.Image(1).subtract(normalizar_imagen(ndvi_p10, geom, "NDVI_P10", ESCALA_MODIS))
    vuln_pend = normalizar_imagen(pendiente, geom, "Slope", ESCALA_DEM)
    iiss = vuln_veg.multiply(0.70).add(vuln_pend.multiply(0.30)).rename("IISS").clip(geom)
    return iiss, ndvi_p10


def clasificar_iiss(iiss, geom):
    """Clasifica IISS en cuatro clases conservando proyección."""
    base = iiss.multiply(0)
    return (
        base
        .where(iiss.gt(0).And(iiss.lte(0.40)), 1)
        .where(iiss.gt(0.40).And(iiss.lte(0.60)), 2)
        .where(iiss.gt(0.60).And(iiss.lte(0.80)), 3)
        .where(iiss.gt(0.80), 4)
        .rename("IISS_clase")
        .clip(geom)
    )


def crear_alerta_integrada(nivel_spi: int, vci_clase, iiss_clase, geom):
    """Integra SPI-3, VCI e IISS asegurando tipado, proyección y máscara limpia para GEE."""
    
    # 1. Usar iiss_clase como base topológica para asegurar la escala correcta
    base = iiss_clase.multiply(0).toByte()
    
    # 2. Forzar spi_img como un entero explícito (Byte) heredando la base
    spi_img = base.add(int(nivel_spi)).toByte().rename("SPI_clase")
    
    # 3. Combinar amenaza (SPI y VCI) evaluando el nivel máximo
    amenaza = spi_img.max(vci_clase).toByte()
    
    # 4. Definir condiciones lógicas con clases de tipo byte
    vigilancia = amenaza.gte(1)
    prealerta = amenaza.gte(2).And(iiss_clase.gte(3))
    alerta = amenaza.gte(3).And(iiss_clase.gte(3))
    emergencia = amenaza.gte(4).And(iiss_clase.eq(4))
    
    # 5. Generar la imagen rasterizada base
    alerta_img = (
        base
        .where(vigilancia, 1)
        .where(prealerta, 2)
        .where(alerta, 3)
        .where(emergencia, 4)
        .toByte()
        .rename("Alerta_Sequia")
    )
    
    # 6. GARANTÍA DE PROYECCIÓN: Forzar escala métrica, enmascarar y recortar (sin .visualize())
    return (
        alerta_img
        .reproject(crs=CRS_METRICO, scale=ESCALA_MODIS)
        .updateMask(alerta_img.gte(0))
        .clip(geom)
    )
    
def area_por_clase(imagen_clase, geom, escala: int = ESCALA_MODIS):
    """Calcula área por clase en hectáreas."""
    area_img = ee.Image.pixelArea().addBands(imagen_clase)
    stats = area_img.reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName="clase"),
        geometry=geom,
        scale=escala,
        maxPixels=1e9,
    )

    registros = []
    for g in stats.getInfo().get("groups", []):
        registros.append({"clase": int(g["clase"]), "area_ha": round(g["sum"] / 10000, 2)})

    df = pd.DataFrame(registros)
    if not df.empty:
        df = df.sort_values("clase")
        df["nivel"] = df["clase"].map(NOMBRES_ALERTA)
        df = df[["clase", "nivel", "area_ha"]]
    return df


# =============================================================================
# MAPAS CON FOLIUM + EARTH ENGINE
# =============================================================================

def obtener_centro_mapa(geom):
    """Obtiene centro de geometría Earth Engine como [lat, lon]."""
    coords = geom.centroid(maxError=1).coordinates().getInfo()
    return [coords[1], coords[0]]

def agregar_capa_ee(mapa, ee_image, vis_params, nombre, opacity=1.0):
    """Agrega una imagen Earth Engine como TileLayer de Folium aplicando visualización segura."""
    img_render = ee.Image(ee_image)
    
    # Si la imagen tiene una sola banda y se pasan parámetros de paleta, aplicamos visualize aquí
    if "palette" in vis_params and img_render.bandNames().size().getInfo() == 1:
        img_render = img_render.visualize(**vis_params)
        # Una vez visualizada en RGB, llamamos a getMapId() sin argumentos adicionales de paleta
        map_id = img_render.getMapId()
    else:
        map_id = img_render.getMapId(vis_params)
    
    folium.raster_layers.TileLayer(
        tiles=map_id["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name=nombre,
        overlay=True,
        control=True,
        opacity=opacity,
    ).add_to(mapa)


def featurecollection_a_imagen(fc, color_value=1, width=2):
    """Convierte un FeatureCollection a imagen de líneas."""
    return ee.Image().byte().paint(featureCollection=fc, color=color_value, width=width)


def construir_grafico_spi(df_spi: pd.DataFrame):
    """Construye gráfico histórico SPI-3."""
    df_plot = df_spi.dropna(subset=["SPI3"]).copy()
    df_plot["color"] = df_plot["SPI3"].apply(lambda x: "#d73027" if x < 0 else "#4575b4")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(df_plot["date"], df_plot["SPI3"], width=25, color=df_plot["color"], alpha=0.75)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(-1.0, color="orange", linestyle="--", label="SPI <= -1.0")
    ax.axhline(-1.5, color="red", linestyle="--", label="SPI <= -1.5")
    ax.axhline(-2.0, color="darkred", linestyle="--", label="SPI <= -2.0")
    ax.set_title("SPI-3 histórico - Microcuenca Amatitán")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("SPI-3")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


# =============================================================================
# APP STREAMLIT
# =============================================================================

st.set_page_config(page_title=APP_TITLE, page_icon="🌱", layout="wide")
st.title("🌱 SAT de Sequía Agrícola - Microcuenca Amatitán")
st.caption("Google Earth Engine + CHIRPS + MODIS + FABDEM + SPI-3 + VCI + IISS")

with st.sidebar:
    st.header("Configuración")
    project_id = st.text_input("Proyecto Google Earth Engine", value=PROJECT_ID_DEFAULT)

    st.subheader("Línea base SPI-3")
    anio_inicio = st.number_input("Año inicial", min_value=1981, max_value=2023, value=ANIO_BASE_SPI_INICIO, step=1)
    anio_fin = st.number_input("Año final", min_value=1990, max_value=ANIO_BASE_SPI_FIN, value=ANIO_BASE_SPI_FIN, step=1)
    st.subheader("MODIS")
    max_retroceso_modis = st.slider("Meses atrás para buscar MODIS", 1, 12, 6)

    st.subheader("Capas del mapa")
    mostrar_alerta = st.checkbox("Alerta integrada", value=True)
    mostrar_iiss = st.checkbox("IISS", value=True)
    mostrar_vci = st.checkbox("VCI actual", value=False)
    mostrar_drenaje = st.checkbox("Drenaje", value=True)

    st.subheader("Umbrales")
    usar_sensibles = st.checkbox("Usar umbrales sensibles de fase piloto", value=False)
    umbrales = UMBRALES_SENSIBLES if usar_sensibles else UMBRALES_ESTANDAR

    if st.button("Limpiar caché y recalcular"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

ok, mensaje = inicializar_gee(project_id)
if not ok:
    st.error(mensaje)
    st.info("En local ejecuta `earthengine authenticate`. En Streamlit Cloud configura `.streamlit/secrets.toml`.")
    st.stop()

st.sidebar.success("GEE conectado")

try:
    microcuenca, geom, drenaje, dem = cargar_assets()
except Exception as exc:
    st.error("No se pudieron cargar los assets. Revisa rutas y permisos en Earth Engine.")
    st.exception(exc)
    st.stop()

with st.spinner("Calculando indicadores del SAT..."):
    try:
        area_km2 = geom.area().divide(1e6).getInfo()
        pendiente = calcular_pendiente(dem, geom)

        df_lluvia = construir_serie_chirps(project_id, int(anio_inicio), int(anio_fin))
        if df_lluvia.empty:
            st.error("No se pudo construir la serie CHIRPS.")
            st.stop()

        df_spi, parametros = calcular_spi3_gamma(df_lluvia)
        spi3_actual, lluvia_3m, mes_ref, fecha_ini, fecha_fin = calcular_spi3_actual(geom, parametros)
        nivel_spi, texto_spi = clasificar_spi3(spi3_actual, umbrales)

        anio_vci, mes_vci = obtener_mes_modis_disponible(geom, max_retroceso_modis)
        vci_actual = calcular_vci_mes(geom, anio_vci, mes_vci)
        vci_prom = promedio_imagen(vci_actual, geom, "VCI", ESCALA_MODIS)
        nivel_vci, texto_vci = clasificar_vci(vci_prom, umbrales)
        vci_clase = clasificar_vci_imagen(vci_actual, geom)

        iiss, ndvi_p10 = calcular_iiss(geom, pendiente)
        iiss_clase = clasificar_iiss(iiss, geom)
        alerta_integrada = crear_alerta_integrada(nivel_spi, vci_clase, iiss_clase, geom)
        df_area = area_por_clase(alerta_integrada, geom, ESCALA_MODIS)
    except Exception as exc:
        st.error("Error durante el cálculo del SAT.")
        st.exception(exc)
        st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Monitoreo", "Mapa", "SPI histórico", "Áreas", "Metodología"])

with tab1:
    st.subheader("Indicadores actuales")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Área", f"{area_km2:.2f} km²")
    col2.metric("SPI-3", "Sin datos" if spi3_actual is None else f"{spi3_actual:.2f}", texto_spi)
    col3.metric("Lluvia 3 meses", "Sin datos" if lluvia_3m is None else f"{lluvia_3m:.1f} mm")
    col4.metric("VCI", "Sin datos" if vci_prom is None else f"{vci_prom:.1f}", texto_vci)
    col5.metric("Estado", NOMBRES_ALERTA.get(max(nivel_spi, nivel_vci), "Sin datos"))

    st.write(f"**Periodo SPI-3:** {fecha_ini} a {fecha_fin}")
    st.write(f"**Mes de referencia VCI:** {'Sin datos' if anio_vci is None else f'{anio_vci}-{mes_vci:02d}'}")

    if nivel_spi >= 4 or nivel_vci >= 4:
        st.error("EMERGENCIA: activar respuesta prioritaria en zonas IISS Muy Alta.")
    elif nivel_spi >= 3 or nivel_vci >= 3:
        st.warning("ALERTA: activar medidas de mitigación en zonas IISS Alta y Muy Alta.")
    elif nivel_spi >= 2 or nivel_vci >= 2:
        st.warning("PREALERTA: iniciar monitoreo quincenal y comunicación preventiva.")
    elif nivel_spi >= 1 or nivel_vci >= 1:
        st.info("VIGILANCIA: mantener monitoreo mensual y revisar pronóstico climático.")
    else:
        st.success("NORMAL: continuar monitoreo rutinario.")

with tab2:
    st.subheader("Mapa integrado de alerta")
   # Reemplazar los valores para quitar el "#"
    vis_alerta = {"min": 0, "max": 4, "palette": [COLORES_ALERTA[i].replace("#", "") for i in range(5)]}
    vis_iiss = {"min": 0, "max": 1, "palette": ["ffffcc", "fed976", "fd8d3c", "fc4e2a", "bd0026", "800026"]}
    vis_vci = {"min": 0, "max": 100, "palette": ["red", "yellow", "green"]} # Los nombres CSS no dan problema

    centro = obtener_centro_mapa(geom)
    mapa = folium.Map(location=centro, zoom_start=13, control_scale=True, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satélite",
        overlay=False,
        control=True,
    ).add_to(mapa)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", overlay=False, control=True).add_to(mapa)

    if mostrar_alerta:
        agregar_capa_ee(mapa, alerta_integrada, vis_alerta, "Alerta integrada", opacity=0.75)
    if mostrar_iiss:
        agregar_capa_ee(mapa, iiss, vis_iiss, "IISS", opacity=0.60)
    if mostrar_vci and vci_actual is not None:
        agregar_capa_ee(mapa, vci_actual, vis_vci, "VCI actual", opacity=0.70)
    if mostrar_drenaje:
        drenaje_img = featurecollection_a_imagen(drenaje, width=2)
        agregar_capa_ee(mapa, drenaje_img, {"palette": ["00ffff"]}, "Drenaje", opacity=1.0)

    limite_img = featurecollection_a_imagen(microcuenca, width=3)
    agregar_capa_ee(mapa, limite_img, {"palette": ["ffffff"]}, "Microcuenca", opacity=1.0)
    folium.LayerControl(collapsed=False).add_to(mapa)
    st_folium(mapa, width=None, height=650)
    st.markdown("**Leyenda:** 🟢 Normal | 🟡 Vigilancia/Prealerta | 🟠 Alerta | 🔴 Emergencia")

with tab3:
    st.subheader("Serie SPI-3")
    st.pyplot(construir_grafico_spi(df_spi))
    with st.expander("Ver tabla SPI-3"):
        st.dataframe(df_spi[["date", "rainfall", "rain_3m", "SPI3"]], use_container_width=True)

with tab4:
    st.subheader("Área por nivel de alerta")
    if df_area.empty:
        st.info("No hay áreas calculadas.")
    else:
        st.dataframe(df_area, use_container_width=True)
        st.download_button(
            "Descargar áreas CSV",
            df_area.to_csv(index=False).encode("utf-8"),
            "areas_alerta_amatitan.csv",
            "text/csv",
        )

with tab5:
    st.subheader("Metodología")
    st.markdown(
        """
        Este geoportal implementa un prototipo de SAT de sequía agrícola para la microcuenca Amatitán.

        - **SPI-3:** indica cuándo existe déficit de lluvia.
        - **VCI:** indica si la vegetación muestra estrés.
        - **IISS:** indica dónde priorizar acciones.
        - **Alerta integrada:** combina clima, vegetación y susceptibilidad espacial.

        El sistema debe validarse con datos locales de campo, calendario agrícola, rendimientos y reportes de afectación.
        """
    )
