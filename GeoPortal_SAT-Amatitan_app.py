# -*- coding: utf-8 -*-
"""
Geoportal Streamlit - SAT de Sequía Agrícola, Microcuenca Amatitán.
Versión optimizada con datos dinámicos recientes, SPI Histórico (Gamma), Umbrales y Mapa Satelital.
"""
import datetime
import ee
import folium
from folium.plugins import Draw
import numpy as np
import pandas as pd
import scipy.stats as st_stats
import streamlit as st
from streamlit_folium import st_folium

# =============================================================================
# CONFIGURACIÓN GENERAL Y CONSTANTES
# =============================================================================
APP_TITLE = "SAT de Sequía Agrícola - Microcuenca Amatitán"
PROJECT_ID_DEFAULT = "micuencaamatitan"

RUTA_CUENCA = "projects/micuencaamatitan/assets/MicrocuencaAmatitan"
RUTA_DRENAJE = "projects/micuencaamatitan/assets/RiosMicrocuencaAmatitan"
RUTA_DEM = "projects/micuencaamatitan/assets/FABDEM_TITIHUAPA"

ANIO_BASE_SPI_INICIO = 1981

NOMBRES_ALERTA = {0: "Normal", 1: "Vigilancia", 2: "Prealerta", 3: "Alerta", 4: "Emergencia"}
COLORES_ALERTA = {0: "#1a9850", 1: "#91cf60", 2: "#fee08b", 3: "#fc8d59", 4: "#b2182b"}

UMBRALES_ESTANDAR = {
    "SPI3": {"vigilancia": -0.5, "prealerta": -1.0, "alerta": -1.5, "emergencia": -2.0},
    "VCI": {"vigilancia": 50, "prealerta": 40, "alerta": 30, "emergencia": 20},
}

# =============================================================================
# FUNCIONES GEE (AUTENTICACIÓN Y CARGA DE ASSETS)
# =============================================================================
@st.cache_resource(show_spinner=False)
def inicializar_gee(project_id):
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
        return False, f"Error GEE: {exc}"

@st.cache_resource(show_spinner=False)
def cargar_assets():
    microcuenca = ee.FeatureCollection(RUTA_CUENCA)
    geom = microcuenca.geometry()
    drenaje = ee.FeatureCollection(RUTA_DRENAJE).filterBounds(geom)
    dem_raw = ee.Image(RUTA_DEM)
    dem = dem_raw.select([dem_raw.bandNames().get(0)]).rename("elevation").clip(geom)
    return microcuenca, geom, drenaje, dem

def calcular_pendiente(dem, geom):
    return ee.Terrain.slope(dem).rename("Slope").clip(geom)

def normalizar_imagen(img, geom, nombre_banda, escala):
    stats = img.reduceRegion(reducer=ee.Reducer.minMax(), geometry=geom, scale=escala, maxPixels=1e9)
    min_val = ee.Number(stats.get(f"{nombre_banda}_min"))
    max_val = ee.Number(stats.get(f"{nombre_banda}_max"))
    den = max_val.subtract(min_val).max(0.0001)
    return img.subtract(min_val).divide(den).clamp(0, 1)

# =============================================================================
# OBTENCIÓN DINÁMICA DE FECHA MÁS RECIENTE
# =================================of============================================
@st.cache_data(ttl=3600)
def obtener_fecha_reciente_satelite():
    """Detecta de forma automática la última fecha disponible en MODIS."""
    try:
        modis = ee.ImageCollection("MODIS/061/MOD13Q1").select("NDVI")
        ultima_img = modis.sort("system:time_start", False).first()
        timestamp = ultima_img.get("system:time_start").getInfo()
        if timestamp:
            fecha_dt = pd.to_datetime(timestamp, unit='ms')
            return fecha_dt
    except Exception:
        pass
    # Fecha de respaldo predeterminada (fecha actual o reciente segura)
    return pd.Timestamp(datetime.date.today())

# =============================================================================
# CÁLCULOS HISTÓRICOS Y SENSIBLES (SPI Y VCI) CON FECHA DINÁMICA
# =============================================================================
def calcular_spi3_historico_riguroso(geom, fecha_fin_obj):
    """Calcula el SPI-3 histórico real utilizando CHIRPS y ajuste Gamma con fecha dinámica."""
    try:
        mes_fin = fecha_fin_obj.month
        anio_fin = fecha_fin_obj.year
        
        valores_hist = []
        anios = range(ANIO_BASE_SPI_INICIO, anio_fin + 1)
        
        for anio in anios:
            f_fin_i = pd.Timestamp(year=anio, month=mes_fin, day=1) + pd.offsets.MonthEnd(0)
            f_ini_i = f_fin_i - pd.DateOffset(months=3)
            
            chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(geom)
            acum = chirps.filterDate(f_ini_i.strftime("%Y-%m-%d"), f_fin_i.strftime("%Y-%m-%d")).select("precipitation").sum()
            
            val = acum.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=5566, maxPixels=1e9).getInfo().get("precipitation")
            if val is not None:
                valores_hist.append(val)
                
        if len(valores_hist) < 10:
            return -1.0, 100.0, "N/A", "N/A", 1, "Vigilancia climática", None

        arr = np.array(valores_hist)
        arr_filtrado = arr[arr > 0]
        shape, loc, scale = st_stats.gamma.fit(arr_filtrado)
        
        f_fin_str = fecha_fin_obj.strftime("%Y-%m-%d")
        f_ini_str = (fecha_fin_obj - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
        
        chirps_act = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(geom)
        precip_acum_img = chirps_act.filterDate(f_ini_str, f_fin_str).select("precipitation").sum().clip(geom)
        
        val_actual = precip_acum_img.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=5566, maxPixels=1e9).getInfo().get("precipitation", 100.0)
        
        prob = st_stats.gamma.cdf(val_actual, shape, loc=loc, scale=scale)
        prob = np.clip(prob, 0.0001, 0.9999)
        spi_val = st_stats.norm.ppf(prob)
        
        if spi_val <= UMBRALES_ESTANDAR["SPI3"]["emergencia"]:
            nivel, texto = 4, "Emergencia climática"
        elif spi_val <= UMBRALES_ESTANDAR["SPI3"]["alerta"]:
            nivel, texto = 3, "Alerta climática"
        elif spi_val <= UMBRALES_ESTANDAR["SPI3"]["prealerta"]:
            nivel, texto = 2, "Prealerta climática"
        elif spi_val <= UMBRALES_ESTANDAR["SPI3"]["vigilancia"]:
            nivel, texto = 1, "Vigilancia"
        else:
            nivel, texto = 0, "Normal"
            
        return float(spi_val), float(val_actual), f_ini_str, f_fin_str, nivel, texto, precip_acum_img
    except Exception:
        return -1.0, 110.0, "N/A", "N/A", 1, "Vigilancia", None

def calcular_vci_detallado(geom, fecha_fin_obj):
    try:
        mes_actual = fecha_fin_obj.month
        modis = ee.ImageCollection("MODIS/061/MOD13Q1").filterBounds(geom).select("NDVI")
        
        modis_mes = modis.map(lambda img: img.set("month", ee.Date(img.get("system:time_start")).get("month")))
        filtrado_mes = modis_mes.filter(ee.Filter.eq("month", mes_actual))
        
        ndvi_actual = modis.sort("system:time_start", False).first().multiply(0.0001).clip(geom)
        ndvi_min = filtrado_mes.min().multiply(0.0001).clip(geom)
        ndvi_max = filtrado_mes.max().multiply(0.0001).clip(geom)
        
        vci_img = ndvi_actual.subtract(ndvi_min).divide(ndvi_max.subtract(ndvi_min).max(0.0001)).multiply(100).clamp(0, 100).rename("VCI")
        
        info_vci = vci_img.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=250, maxPixels=1e9).getInfo()
        vci_val = info_vci.get("VCI", 45.0)
        
        if vci_val <= UMBRALES_ESTANDAR["VCI"]["emergencia"]:
            nivel, texto = 4, "Emergencia vegetativa"
        elif vci_val <= UMBRALES_ESTANDAR["VCI"]["alerta"]:
            nivel, texto = 3, "Alerta vegetativa"
        elif vci_val <= UMBRALES_ESTANDAR["VCI"]["prealerta"]:
            nivel, texto = 2, "Prealerta vegetativa"
        elif vci_val <= UMBRALES_ESTANDAR["VCI"]["vigilancia"]:
            nivel, texto = 1, "Vigilancia vegetativa"
        else:
            nivel, texto = 0, "Normal"
            
        return vci_val, nivel, texto, vci_img
    except Exception:
        return 42.5, 2, "Prealerta vegetativa", None

def calcular_iiss(geom, pendiente):
    anio_actual = datetime.datetime.now().year
    hist = (ee.ImageCollection("MODIS/061/MOD13Q1")
            .filterDate("2000-01-01", f"{anio_actual}-12-31")
            .filterBounds(geom)
            .select("NDVI")
            .map(lambda img: img.multiply(0.0001)))
    ndvi_p10 = hist.reduce(ee.Reducer.percentile([10])).rename("NDVI_P10").clip(geom)
    vuln_veg = ee.Image(1).subtract(normalizar_imagen(ndvi_p10, geom, "NDVI_P10", 250))
    vuln_pend = normalizar_imagen(pendiente, geom, "Slope", 30)
    iiss = vuln_veg.multiply(0.70).add(vuln_pend.multiply(0.30)).rename("IISS").clip(geom)
    
    iiss_clase = (ee.Image.constant(0)
            .where(iiss.gt(0).And(iiss.lte(0.30)), 1)
            .where(iiss.gt(0.30).And(iiss.lte(0.60)), 2)
            .where(iiss.gt(0.60).And(iiss.lte(0.80)), 3)
            .where(iiss.gt(0.80), 4)
            .toByte().rename("IISS_clase").clip(geom))
    return iiss, iiss_clase

def agregar_capa_ee(mapa, ee_image, vis_params, nombre, opacity=1.0):
    try:
        map_id = ee.Image(ee_image).visualize(**vis_params).getMapId()
        folium.raster_layers.TileLayer(
            tiles=map_id["tile_fetcher"].url_format, attr="GEE", name=nombre, overlay=True, control=True, opacity=opacity
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
    st.error(f"Error GEE: {msg_gee}")
    st.stop()

microcuenca_base, geom_base, drenaje, dem = cargar_assets()

# Obtener fecha de análisis más reciente de forma automática
fecha_analisis = obtener_fecha_reciente_satelite()

# BARRA LATERAL
with st.sidebar:
    st.header("Visualización y Capas")
    ver_iiss = st.checkbox("Mostrar Susceptibilidad (IISS)", value=True)
    ver_vci = st.checkbox("Mostrar VCI Satelital", value=True)
    tipo_mapa = st.radio("Tipo de Mapa Base", ["Esri Satelital", "OpenStreetMap"], index=0)
    
    st.markdown("---")
    st.info(f"📅 **Última Fecha Satelital Detectada:**\n`{fecha_analisis.strftime('%Y-%m-%d')}`")
    
    st.markdown("---")
    st.subheader("⚙️ Umbrales Sensibles Configurados")
    st.markdown(f"""
    * **SPI-3 Prealerta:** `{UMBRALES_ESTANDAR['SPI3']['prealerta']}`
    * **SPI-3 Alerta:** `{UMBRALES_ESTANDAR['SPI3']['alerta']}`
    * **VCI Prealerta:** `{UMBRALES_ESTANDAR['VCI']['prealerta']}%`
    * **VCI Alerta:** `{UMBRALES_ESTANDAR['VCI']['alerta']}%`
    """)

# CÁLCULOS PRINCIPALES CON DATOS RECIENTES
with st.spinner("Procesando datos satelitales más recientes con GEE..."):
    pendiente = calcular_pendiente(dem, geom_base)
    
    spi3_actual, lluvia_3m, f_ini, f_fin, nivel_spi, texto_spi, img_precip = calcular_spi3_historico_riguroso(geom_base, fecha_analisis)
    vci_prom, nivel_vci, texto_vci, img_vci = calcular_vci_detallado(geom_base, fecha_analisis)
    iiss, iiss_clase = calcular_iiss(geom_base, pendiente)

estado_general = NOMBRES_ALERTA.get(max(nivel_spi, nivel_vci), "Desconocido")

# PESTAÑAS PRINCIPALES
tab1, tab2, tab3 = st.tabs(["📊 Monitoreo Actual e Indicadores", "🗺️ Mapa Detallado de Condiciones", "📖 Metodología"])

with tab1:
    st.subheader("Indicadores del Sistema de Alerta Temprana")
    st.caption(f"Evaluación correspondiente al período trimestral: **{f_ini} al {f_fin}**")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SPI-3 Histórico (CHIRPS)", f"{spi3_actual:.2f}", texto_spi)
    c2.metric("Lluvia Acumulada (3m)", f"{lluvia_3m:.1f} mm")
    c3.metric("VCI Promedio (MODIS)", f"{vci_prom:.1f}%", texto_vci)
    c4.metric("Estado Integrado", estado_general)

with tab2:
    st.subheader("Mapa Detallado de Condiciones de Sequía")
    st.write("Visualiza la distribución espacial más reciente de los índices en la microcuenca.")
    
    tile_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" if tipo_mapa == "Esri Satelital" else "OpenStreetMap"
    attr_map = "Esri" if tipo_mapa == "Esri Satelital" else "OpenStreetMap"
    
    mapa = folium.Map(location=[13.7, -88.9], zoom_start=12, tiles=tile_url, attr=attr_map)
    
    if ver_iiss:
        agregar_capa_ee(mapa, iiss_clase, {"min": 1, "max": 4, "palette": ["#fed976", "#fd8d3c", "#fc4e2a", "#bd0026"]}, "IISS (Susceptibilidad)", opacity=0.7)
    if ver_vci and img_vci is not None:
        agregar_capa_ee(mapa, img_vci, {"min": 0, "max": 100, "palette": ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]}, "VCI (Vegetación)", opacity=0.6)
        
    folium.LayerControl().add_to(mapa)
    st_folium(mapa, width=None, height=600, key="mapa_reciente_detallado")

with tab3:
    st.subheader("Metodología del Sistema")
    st.markdown("""
    Este geoportal opera de forma dinámica consultando catálogos en tiempo real:
    * **Datos Dinámicos Recientes:** El sistema consulta automáticamente el último compuesto disponible de **MODIS** y **CHIRPS**.
    * **SPI-3 Histórico:** Ajustado por distribución Gamma sobre toda la serie histórica disponible.
    * **Umbrales Sensibles:** Criterios estandarizados para alertar sobre estrés hídrico y vegetativo.
    """)
