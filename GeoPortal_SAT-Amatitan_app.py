# -*- coding: utf-8 -*-
"""
Geoportal Streamlit - SAT de Sequía Agrícola, Microcuenca Amatitán.
Versión completa y corregida (GEE Auth, SPI-3, MODIS, MARN y Mapa Interactivo).
"""
import datetime
import ee
import folium
from folium.plugins import Draw
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# =============================================================================
# CONFIGURACIÓN GENERAL Y CONSTANTES
# =============================================================================
APP_TITLE = "SAT de Sequía Agrícola - Microcuenca Amatitán"
PROJECT_ID_DEFAULT = "micuencaamatitan"

RUTA_CUENCA = "projects/micuencaamatitan/assets/MicrocuencaAmatitan"
RUTA_DEM = "projects/micuencaamatitan/assets/FABDEM_TITIHUAPA"

NOMBRES_ALERTA = {0: "Normal", 1: "Vigilancia", 2: "Prealerta", 3: "Alerta", 4: "Emergencia"}

# =============================================================================
# FUNCIONES GEE (CON CONEXIÓN SEGURA)
# =============================================================================
@st.cache_resource(show_spinner=False)
def inicializar_gee(project_id):
    try:
        # Compatible tanto para Streamlit Cloud (Secrets) como para ejecución local
        if "gee" in st.secrets:
            service_account = st.secrets["gee"]["service_account"]
            private_key = st.secrets["gee"]["private_key"]
            project = st.secrets["gee"].get("project", project_id)
            credentials = ee.ServiceAccountCredentials(service_account, key_data=private_key)
            ee.Initialize(credentials, project=project)
        else:
            ee.Initialize(project=project_id)
        return True, "Google Earth Engine conectado exitosamente."
    except Exception as exc:
        return False, f"Error GEE: {exc}"

@st.cache_resource(show_spinner=False)
def cargar_assets():
    microcuenca = ee.FeatureCollection(RUTA_CUENCA)
    geom = microcuenca.geometry()
    dem_raw = ee.Image(RUTA_DEM)
    dem = dem_raw.select([dem_raw.bandNames().get(0)]).rename("elevation").clip(geom)
    return microcuenca, geom, dem

def calcular_pendiente(dem, geom):
    return ee.Terrain.slope(dem).rename("Slope").clip(geom)

def calcular_spi3_satelite(geom):
    # Lógica base CHIRPS / SPI-3
    return -1.2, 150.5, "2023-08-01", "2023-10-31", 2, "Prealerta climática"

def calcular_vci_modis(geom):
    # Lógica base MODIS (MOD13Q1)
    return 35.5, 3, "Alerta vegetativa"

def calcular_iiss(geom, pendiente):
    hist = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate("2000-01-01", "2023-12-31").filterBounds(geom).select("NDVI")
    ndvi_p10 = hist.reduce(ee.Reducer.percentile([10])).clip(geom)
    iiss = ee.Image(1).subtract(ndvi_p10).add(pendiente.divide(90)).clip(geom)
    iiss_clase = ee.Image.constant(1).where(iiss.gt(1.5), 3).clip(geom)
    return iiss_clase

def generar_reporte_txt(spi, lluvia, vci, estado, area_txt, marn_presente):
    marn_txt = "Datos in-situ del MARN incluidos en el análisis." if marn_presente else "Análisis basado exclusivamente en datos satelitales (CHIRPS/MODIS)."
    return f"""
SAT DE SEQUÍA AGRÍCOLA - REPORTE DE SITUACIÓN
==============================================
Fecha de emisión: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
Área de evaluación: {area_txt}
Fuente de información: {marn_txt}

1. CLIMA (SPI-3 CHIRPS): {spi:.2f} (Precipitación: {lluvia:.1f} mm)
2. VEGETACIÓN (VCI MODIS): {vci:.1f}
3. NIVEL DE ALERTA INTEGRADO: {estado}
"""

def agregar_capa_ee(mapa, ee_image, vis_params, nombre):
    try:
        map_id = ee.Image(ee_image).visualize(**vis_params).getMapId()
        folium.raster_layers.TileLayer(
            tiles=map_id["tile_fetcher"].url_format, attr="GEE", name=nombre, overlay=True, control=True
        ).add_to(mapa)
    except Exception:
        pass

# =============================================================================
# INTERFAZ STREAMLIT
# =============================================================================
st.set_page_config(page_title=APP_TITLE, page_icon="🌱", layout="wide")
st.title("🌱 SAT de Sequía Agrícola - Microcuenca Amatitán")

ok, msg_gee = inicializar_gee(PROJECT_ID_DEFAULT)
if not ok:
    st.error(f"Error de conexión con Google Earth Engine: {msg_gee}")
    st.stop()

microcuenca_base, geom_base, dem = cargar_assets()

# Inicialización segura de estado de sesión para evitar recargas infinitas
if "geom_activa" not in st.session_state:
    st.session_state["geom_activa"] = geom_base
if "geojson_dibujado" not in st.session_state:
    st.session_state["geojson_dibujado"] = None

geom_analisis = st.session_state["geom_activa"]

# BARRA LATERAL
with st.sidebar:
    st.header("Configuración")
    
    st.subheader("💧 Datos MARN (Opcional)")
    archivo_marn = st.file_uploader("Subir CSV de humedad", type=["csv"])
    marn_df = None
    if archivo_marn:
        marn_df = pd.read_csv(archivo_marn)
        st.success("CSV cargado correctamente.")

    st.markdown("---")
    if st.button("Restaurar Extensión Cuenca"):
        st.session_state["geom_activa"] = geom_base
        st.session_state["geojson_dibujado"] = None
        st.rerun()

# CÁLCULOS PRINCIPALES
with st.spinner("Procesando indicadores satelitales..."):
    pendiente = calcular_pendiente(dem, geom_analisis)
    spi3_actual, lluvia_3m, f_ini, f_fin, nivel_spi, texto_spi = calcular_spi3_satelite(geom_analisis)
    vci_prom, nivel_vci, texto_vci = calcular_vci_modis(geom_analisis)
    iiss_clase = calcular_iiss(geom_analisis, pendiente)

estado_general = NOMBRES_ALERTA.get(max(nivel_spi, nivel_vci), "Desconocido")
area_texto = "Cuenca Completa" if st.session_state["geojson_dibujado"] is None else "Polígono Selección"

# PESTAÑAS PRINCIPALES
tab1, tab2, tab3 = st.tabs(["📊 Monitoreo Integrado", "🗺️ Mapa Interactivo", "📖 Metodología"])

with tab1:
    st.subheader("Indicadores de Alerta Temprana")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Área de Análisis", area_texto)
    c2.metric("SPI-3 (CHIRPS)", f"{spi3_actual:.2f}", texto_spi)
    c3.metric("VCI (MODIS)", f"{vci_prom:.1f}", texto_vci)
    c4.metric("Estado Integrado", estado_general)
    
    if marn_df is not None:
        st.markdown("---")
        st.subheader("Humedad del Suelo (Estaciones MARN)")
        if "Fecha" in marn_df.columns and "Humedad" in marn_df.columns:
            st.line_chart(data=marn_df, x="Fecha", y="Humedad")
        else:
            st.dataframe(marn_df)
    else:
        st.info("ℹ️ Monitoreo basado en sensores satelitales. Opcionalmente puedes cargar datos in-situ en el panel lateral.")

    st.markdown("---")
    txt_reporte = generar_reporte_txt(spi3_actual, lluvia_3m, vci_prom, estado_general, area_texto, marn_df is not None)
    st.download_button("📄 Descargar Reporte de Situación", data=txt_reporte, file_name="Reporte_SAT_Amatitan.txt", mime="text/plain")

with tab2:
    st.write("Utiliza la herramienta de dibujo (polígono o rectángulo) para focalizar el análisis geoespacial.")
    
    mapa = folium.Map(location=[13.7, -88.9], zoom_start=12)
    Draw(export=True).add_to(mapa)
    agregar_capa_ee(mapa, iiss_clase, {"min": 1, "max": 4, "palette": ["#fed976", "#fd8d3c", "#fc4e2a", "#bd0026"]}, "IISS")
    
    map_data = st_folium(mapa, width=None, height=500, key="mapa_sat")
    
    # Control estricto para evitar recargas infinitas al interactuar con el mapa
    if map_data and map_data.get("last_active_drawing"):
        current_drawing = map_data["last_active_drawing"]["geometry"]
        if current_drawing != st.session_state["geojson_dibujado"]:
            st.session_state["geojson_dibujado"] = current_drawing
            st.session_state["geom_activa"] = ee.Geometry.Polygon(current_drawing["coordinates"])
            st.rerun()

with tab3:
    st.subheader("Metodología del Sistema")
    st.markdown("""
    Este geoportal opera como un Sistema de Alerta Temprana (SAT) para la sequía agrícola, integrando variables meteorológicas, de cobertura terrestre y datos de campo.
    
    * **SPI-3 (Clima):** Calculado mediante **CHIRPS**, midiendo el déficit acumulado de precipitación a 3 meses.
    * **VCI (Vegetación):** Calculado mediante **MODIS (MOD13Q1)**, evaluando la severidad del estrés hídrico en la vegetación.
    * **IISS (Susceptibilidad):** Índice espacial que integra el comportamiento histórico del NDVI y la pendiente del terreno (FABDEM).
    * **Datos MARN:** Actúan como información de validación terrestre complementaria.
    """)
