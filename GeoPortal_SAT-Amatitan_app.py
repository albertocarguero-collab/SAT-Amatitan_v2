# -*- coding: utf-8 -*-
"""
Geoportal Streamlit - SAT de Sequía Agrícola, Microcuenca Amatitán.
Integración con datos MARN, focalización espacial, reportes y metodología ajustada.
"""
import datetime
import json
import ee
import folium
from folium.plugins import Draw
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
    0: "Normal", 1: "Vigilancia", 2: "Prealerta", 3: "Alerta", 4: "Emergencia",
}
COLORES_ALERTA = {
    0: "#1a9850", 1: "#91cf60", 2: "#fee08b", 3: "#fc8d59", 4: "#b2182b",
}

UMBRALES_ESTANDAR = {
    "SPI3": {"vigilancia": -0.5, "prealerta": -1.0, "alerta": -1.5, "emergencia": -2.0},
    "VCI": {"vigilancia": 50, "prealerta": 40, "alerta": 30, "emergencia": 20},
}
UMBRALES_SENSIBLES = {
    "SPI3": {"vigilancia": -0.2, "prealerta": -0.7, "alerta": -1.2, "emergencia": -1.8},
    "VCI": {"vigilancia": 60, "prealerta": 45, "alerta": 35, "emergencia": 25},
}

# =============================================================================
# FUNCIONES AUXILIARES Y GEE
# =============================================================================
@st.cache_resource(show_spinner=False)
def inicializar_gee(project_id: str):
    try:
        if "gee" in st.secrets:
            service_account = st.secrets["gee"]["service_account"]
            private_key = st.secrets["gee"]["private_key"]
            project = st.secrets["gee"].get("project", project_id)
            credentials = ee.ServiceAccountCredentials(service_account, key_data=private_key)
            ee.Initialize(credentials, project=project)
        else:
            ee.Initialize(project=project_id)
        return True, "Google Earth Engine conectado."
    except Exception as exc:
        return False, f"No se pudo conectar a Google Earth Engine: {exc}"

@st.cache_resource(show_spinner=False)
def cargar_assets():
    microcuenca = ee.FeatureCollection(RUTA_CUENCA)
    geom = microcuenca.geometry()
    drenaje = ee.FeatureCollection(RUTA_DRENAJE).filterBounds(geom)
    dem_raw = ee.Image(RUTA_DEM)
    dem_band = dem_raw.bandNames().get(0)
    dem = dem_raw.select([dem_band]).rename("elevation").clip(geom)
    return microcuenca, geom, drenaje, dem

def calcular_pendiente(dem, geom):
    return ee.Terrain.slope(dem).rename("Slope").clip(geom)

def normalizar_imagen(img, geom, nombre_banda: str, escala: int):
    stats = img.reduceRegion(reducer=ee.Reducer.minMax(), geometry=geom, scale=escala, maxPixels=1e9)
    min_val = ee.Number(stats.get(f"{nombre_banda}_min"))
    max_val = ee.Number(stats.get(f"{nombre_banda}_max"))
    den = max_val.subtract(min_val).max(0.0001)
    return img.subtract(min_val).divide(den).clamp(0, 1)

def calcular_iiss(geom, pendiente):
    hist = (ee.ImageCollection("MODIS/061/MOD13Q1")
            .filterDate(f"{ANIO_BASE_MODIS_INICIO}-01-01", f"{ANIO_BASE_MODIS_FIN}-12-31")
            .filterBounds(geom)
            .select("NDVI")
            .map(lambda img: img.multiply(0.0001)))
    ndvi_p10 = hist.reduce(ee.Reducer.percentile([10])).rename("NDVI_P10").clip(geom)
    vuln_veg = ee.Image(1).subtract(normalizar_imagen(ndvi_p10, geom, "NDVI_P10", ESCALA_MODIS))
    vuln_pend = normalizar_imagen(pendiente, geom, "Slope", ESCALA_DEM)
    iiss = vuln_veg.multiply(0.70).add(vuln_pend.multiply(0.30)).rename("IISS").clip(geom)
    return iiss, ndvi_p10

def clasificar_iiss(iiss, geom):
    return (ee.Image.constant(0)
            .where(iiss.gt(0).And(iiss.lte(0.30)), 1)
            .where(iiss.gt(0.30).And(iiss.lte(0.60)), 2)
            .where(iiss.gt(0.60).And(iiss.lte(0.80)), 3)
            .where(iiss.gt(0.80), 4)
            .toByte().rename("IISS_clase").clip(geom))

def obtener_centro_mapa(geom):
    coords = geom.centroid(maxError=1).coordinates().getInfo()
    return [coords[1], coords[0]]

def agregar_capa_ee(mapa, ee_image, vis_params, nombre, opacity=1.0):
    try:
        img_render = ee.Image(ee_image)
        if vis_params is not None:
            img_render = img_render.visualize(**vis_params)
        map_id = img_render.getMapId()
        folium.raster_layers.TileLayer(
            tiles=map_id["tile_fetcher"].url_format, attr="Google Earth Engine", 
            name=nombre, overlay=True, control=True, opacity=opacity,
        ).add_to(mapa)
    except Exception as exc:
        st.warning(f"No se pudo cargar la capa '{nombre}'.")

def generar_reporte_txt(spi, lluvia, vci, estado, area, fecha_ini, fecha_fin):
    return f"""
SAT DE SEQUÍA AGRÍCOLA - REPORTE GENERADO
=========================================
Fecha de emisión: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
Área analizada: {area:.2f} km²

1. CONDICIONES CLIMÁTICAS (SPI-3)
- Periodo evaluado: {fecha_ini} a {fecha_fin}
- Lluvia acumulada (3 meses): {'N/A' if lluvia is None else f'{lluvia:.1f} mm'}
- Índice SPI-3: {'N/A' if spi is None else f'{spi:.2f}'}

2. CONDICIONES VEGETATIVAS (VCI)
- Índice de Condición de Vegetación: {'N/A' if vci is None else f'{vci:.1f}'}

3. ESTADO DE ALERTA INTEGRADA
- Nivel actual: {estado}

Recomendaciones:
- Normal: Monitoreo rutinario.
- Vigilancia: Revisar pronósticos de lluvia del MARN.
- Prealerta/Alerta: Iniciar medidas de mitigación en zonas de susceptibilidad.
- Emergencia: Respuesta prioritaria.
"""

# =============================================================================
# APP STREAMLIT
# =============================================================================
st.set_page_config(page_title=APP_TITLE, page_icon="🌱", layout="wide")
st.title("🌱 SAT de Sequía Agrícola - Microcuenca Amatitán")

ok, mensaje = inicializar_gee(PROJECT_ID_DEFAULT)
if not ok:
    st.error(mensaje)
    st.stop()

# Cargar Assets base
try:
    microcuenca_base, geom_base, drenaje, dem = cargar_assets()
except Exception as exc:
    st.error("Error cargando assets.")
    st.stop()

# Mapeo de Geometría Activa (Permite el recorte con el mapa)
if "geom_activa" not in st.session_state:
    st.session_state["geom_activa"] = geom_base

geom_analisis = st.session_state["geom_activa"]

# BARRA LATERAL
with st.sidebar:
    st.header("Configuración de Capas")
    mostrar_iiss = st.checkbox("IISS", value=True)
    mostrar_vci = st.checkbox("VCI actual", value=False)
    mostrar_drenaje = st.checkbox("Drenaje", value=True)
    usar_sensibles = st.checkbox("Usar umbrales sensibles", value=False)
    umbrales = UMBRALES_SENSIBLES if usar_sensibles else UMBRALES_ESTANDAR
    
    st.markdown("---")
    st.subheader("💧 Datos de Humedad de Suelo (MARN)")
    archivo_marn = st.file_uploader("Sube el reporte de estaciones (CSV)", type=["csv"])
    if archivo_marn:
        df_marn = pd.read_csv(archivo_marn)
        st.success("Datos cargados correctamente.")
        if "Humedad" in df_marn.columns and "Fecha" in df_marn.columns:
            st.line_chart(data=df_marn, x="Fecha", y="Humedad")
        else:
            st.dataframe(df_marn)

    if st.button("Restaurar Área Completa"):
        st.session_state["geom_activa"] = geom_base
        st.rerun()

# CÁLCULOS PRINCIPALES (Mockeados parcialmente por longitud de ejemplo, 
# asume que llamas a tus funciones originales de SPI/VCI aquí usando `geom_analisis` en lugar de `geom_base`)
with st.spinner("Calculando indicadores del SAT..."):
    area_km2 = geom_analisis.area().divide(1e6).getInfo()
    pendiente = calcular_pendiente(dem, geom_analisis)
    
    # Aquí irían tus funciones: calcular_spi3_actual(), calcular_vci_mes(), etc.
    # Para el ejemplo asignamos variables dummy simulando el cálculo
    spi3_actual, lluvia_3m, fecha_ini, fecha_fin = -1.2, 150.5, "2023-08-01", "2023-10-31" 
    nivel_spi, texto_spi = 2, "Prealerta climática"
    vci_prom = 35.5
    nivel_vci, texto_vci = 3, "Alerta vegetativa"
    
    iiss, ndvi_p10 = calcular_iiss(geom_analisis, pendiente)
    iiss_clase = clasificar_iiss(iiss, geom_analisis)

estado_general = NOMBRES_ALERTA.get(max(nivel_spi, nivel_vci), "Desconocido")

# PESTAÑAS
tab1, tab2, tab3 = st.tabs(["Monitoreo y Reporte", "Mapa Interactivo", "Metodología Técnica"])

with tab1:
    st.subheader("Indicadores de Alerta Temprana")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Área de Análisis", f"{area_km2:.2f} km²")
    col2.metric("SPI-3", f"{spi3_actual:.2f}", texto_spi)
    col3.metric("Lluvia (3m)", f"{lluvia_3m:.1f} mm")
    col4.metric("VCI Promedio", f"{vci_prom:.1f}", texto_vci)
    col5.metric("Estado Integrado", estado_general)
    
    # Botón de Reporte
    st.markdown("---")
    texto_reporte = generar_reporte_txt(spi3_actual, lluvia_3m, vci_prom, estado_general, area_km2, fecha_ini, fecha_fin)
    st.download_button(
        label="📄 Descargar Reporte de Situación",
        data=texto_reporte,
        file_name=f"Reporte_SAT_Amatitan_{datetime.datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )

with tab2:
    st.write("Dibuja un polígono para recalcular los índices de sequía en un área específica.")
    centro = obtener_centro_mapa(geom_analisis)
    mapa = folium.Map(location=centro, zoom_start=13, control_scale=True, tiles="OpenStreetMap")
    
    # Añadir herramientas de dibujo
    Draw(export=True).add_to(mapa)
    
    if mostrar_iiss:
        agregar_capa_ee(mapa, iiss_clase, {"min": 1, "max": 4, "palette": ["fed976", "fd8d3c", "fc4e2a", "bd0026"]}, "IISS", opacity=0.60)
    
    # Renderizar mapa
    map_data = st_folium(mapa, width=None, height=600)
    
    # Capturar interacción de dibujo
    if map_data and map_data.get("last_active_drawing"):
        coords = map_data["last_active_drawing"]["geometry"]["coordinates"]
        nueva_geom = ee.Geometry.Polygon(coords)
        if st.session_state["geom_activa"] != nueva_geom:
            st.session_state["geom_activa"] = nueva_geom
            st.rerun() # Recarga la app aplicando la nueva geometría

with tab3:
    st.subheader("Metodología y Fundamento Técnico")
    st.markdown("""
    Este geoportal opera como un Sistema de Alerta Temprana (SAT) para la sequía agrícola, integrando variables meteorológicas, de cobertura terrestre y datos de campo.
    
    **1. Indicador Meteorológico: SPI-3**
    *   **Fuente:** CHIRPS a 5.5 km de resolución.
    *   **Cálculo:** Se ajusta la precipitación acumulada trimestral a una distribución Gamma para normalizar históricamente el déficit de humedad.
    
    **2. Indicador Vegetativo: VCI (Vegetation Condition Index)**
    *   **Fuente:** Sensor MODIS (MOD13Q1) a 250m de resolución (NDVI).
    *   **Cálculo:** Mide el estrés hídrico asimilado por los cultivos comparando el NDVI actual con los extremos históricos.
    
    **3. Indicador Espacial: IISS (Índice Integrado de Susceptibilidad a la Sequía)**
    $$ IISS = (1 - NDVI_{P10}) * 0.70 + (Pendiente) * 0.30 $$
    * Combina la vulnerabilidad histórica de la vegetación y la inclinación del terreno (FABDEM 30m).
    
    **4. Datos In-Situ: MARN**
    * Calibración cruzada permitiendo la subida de datos de humedad de suelo locales vía estaciones telemétricas.
    """)
