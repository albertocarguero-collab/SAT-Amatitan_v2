# app.py
# ==============================================================================
# Geoportal Streamlit - SAT de Sequia Agricola Microcuenca Amatitan
# Enfoque: Google Earth Engine + CHIRPS + MODIS VCI + FABDEM + IISS + Alerta integrada
# Autor: Carlos Carbajal / Curso MCHV-513
# ==============================================================================

import datetime
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

import streamlit as st
import ee
import geemap.foliumap as geemap
from streamlit_folium import st_folium


# ==============================================================================
# CONFIGURACION DE LA APP
# ==============================================================================

st.set_page_config(
    page_title="SAT Sequia Agricola Amatitan",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==============================================================================
# PARAMETROS GENERALES
# ==============================================================================

DEFAULT_PROJECT_ID = "micuencaamatitan"
RUTA_CUENCA = "projects/micuencaamatitan/assets/MicrocuencaAmatitan"
RUTA_DRENAJE = "projects/micuencaamatitan/assets/RiosMicrocuencaAmatitan"
RUTA_DEM = "projects/micuencaamatitan/assets/FABDEM_TITIHUAPA"

CRS_METRICO = "EPSG:32616"  # UTM Zona 16N, adecuada para El Salvador
ESCALA_DEM = 30
ESCALA_CHIRPS = 5566
ESCALA_MODIS = 250

NOMBRES_ALERTA = {
    0: "Normal",
    1: "Vigilancia",
    2: "Prealerta",
    3: "Alerta",
    4: "Emergencia"
}

COLORES_ALERTA = {
    0: "#1a9850",
    1: "#91cf60",
    2: "#fee08b",
    3: "#fc8d59",
    4: "#b2182b"
}

UMBRALES = {
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
    "IISS": {
        "baja_max": 0.40,
        "moderada_max": 0.60,
        "alta_max": 0.80,
        "muy_alta_min": 0.80,
    },
}


# ==============================================================================
# INICIALIZACION DE GOOGLE EARTH ENGINE
# ==============================================================================

@st.cache_resource(show_spinner=False)
def inicializar_gee(project_id: str):
    """
    Inicializa Google Earth Engine.

    Opciones:
    1. Local: usar `earthengine authenticate` antes de ejecutar Streamlit.
    2. Streamlit Cloud: configurar una cuenta de servicio en st.secrets["gee"].
    """
    try:
        if "gee" in st.secrets:
            service_account = st.secrets["gee"].get("service_account")
            private_key = st.secrets["gee"].get("private_key")
            project = st.secrets["gee"].get("project", project_id)

            credentials = ee.ServiceAccountCredentials(
                service_account,
                key_data=private_key
            )
            ee.Initialize(credentials, project=project)
        else:
            ee.Initialize(project=project_id)
        return True, "Google Earth Engine conectado correctamente."

    except Exception as e:
        # En ambiente local se puede intentar autenticacion interactiva.
        # En Streamlit Cloud esto no funciona, por eso se informa con claridad.
        try:
            ee.Authenticate()
            ee.Initialize(project=project_id)
            return True, "Google Earth Engine autenticado e inicializado."
        except Exception as e2:
            return False, f"No se pudo inicializar Google Earth Engine: {e2}"


# ==============================================================================
# CARGA DE ASSETS
# ==============================================================================

@st.cache_resource(show_spinner=False)
def cargar_assets():
    """
    Carga microcuenca, red de drenaje y DEM desde los assets de Earth Engine.
    """
    microcuenca = ee.FeatureCollection(RUTA_CUENCA)
    geom = microcuenca.geometry()

    drenaje = ee.FeatureCollection(RUTA_DRENAJE).filterBounds(geom)

    dem_raw = ee.Image(RUTA_DEM)
    dem_band = dem_raw.bandNames().get(0)
    dem = dem_raw.select([dem_band]).rename("elevation").clip(geom)

    return microcuenca, geom, drenaje, dem


# ==============================================================================
# FUNCIONES BASE DEL SAT
# ==============================================================================

def calcular_pendiente(dem, geom):
    """
    Calcula pendiente en grados usando una proyeccion metrica.
    """
    dem_metric = dem.reproject(crs=CRS_METRICO, scale=ESCALA_DEM)
    pendiente = ee.Terrain.slope(dem_metric).rename("Slope").clip(geom)
    return pendiente


def lluvia_mensual_feature(year, month, geom):
    """
    Calcula lluvia mensual promedio CHIRPS dentro de la microcuenca.
    """
    inicio = ee.Date.fromYMD(year, month, 1)
    fin = inicio.advance(1, "month")

    lluvia_mes = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(inicio, fin)
        .select("precipitation")
        .sum()
        .clip(geom)
    )

    lluvia_media = lluvia_mes.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=ESCALA_CHIRPS,
        maxPixels=1e9,
    ).get("precipitation")

    return ee.Feature(
        None,
        {
            "date": inicio.format("YYYY-MM-dd"),
            "year": year,
            "month": month,
            "rainfall": lluvia_media,
        },
    )


@st.cache_data(show_spinner=False)
def construir_serie_chirps_cached(project_id: str, anio_inicio: int, anio_fin: int):
    """
    Construye serie mensual CHIRPS y la guarda en cache de Streamlit.
    Se vuelve a cargar la geometria dentro de la funcion para evitar problemas
    de hash con objetos de Earth Engine.
    """
    ok, msg = inicializar_gee(project_id)
    if not ok:
        raise RuntimeError(msg)

    microcuenca, geom, drenaje, dem = cargar_assets()

    features = []
    for year in range(anio_inicio, anio_fin + 1):
        for month in range(1, 13):
            features.append(lluvia_mensual_feature(year, month, geom))

    fc = ee.FeatureCollection(features)
    datos = fc.getInfo()["features"]

    df = pd.DataFrame([f["properties"] for f in datos])
    df["date"] = pd.to_datetime(df["date"])
    df["rainfall"] = pd.to_numeric(df["rainfall"], errors="coerce")
    df = df.dropna(subset=["rainfall"]).sort_values("date").reset_index(drop=True)

    return df


def calcular_spi3_gamma(df_lluvia):
    """
    Calcula SPI-3 con ajuste Gamma por mes calendario.
    """
    df = df_lluvia.copy().sort_values("date").reset_index(drop=True)

    df["rain_3m"] = df["rainfall"].rolling(window=3, min_periods=3).sum()
    df["month_end"] = df["date"].dt.month
    df["SPI3"] = np.nan

    parametros_gamma = {}

    for mes in range(1, 13):
        idx = df["month_end"] == mes
        valores = df.loc[idx, "rain_3m"].dropna()

        if len(valores) < 10:
            continue

        n_total = len(valores)
        n_ceros = (valores <= 0).sum()
        prob_cero = n_ceros / n_total
        valores_pos = valores[valores > 0]

        if len(valores_pos) < 10:
            continue

        shape, loc, scale = st.gamma.fit(valores_pos, floc=0)
        parametros_gamma[mes] = {
            "shape": shape,
            "loc": loc,
            "scale": scale,
            "prob_cero": prob_cero,
        }

        rain_values = df.loc[idx, "rain_3m"]
        cdf_gamma = st.gamma.cdf(rain_values, shape, loc=loc, scale=scale)
        cdf_ajustada = prob_cero + (1 - prob_cero) * cdf_gamma
        cdf_ajustada = np.clip(cdf_ajustada, 0.0001, 0.9999)
        df.loc[idx, "SPI3"] = st.norm.ppf(cdf_ajustada)

    return df, parametros_gamma


def calcular_spi3_actual(geom, parametros_gamma):
    """
    Calcula SPI-3 actual con los ultimos 3 meses completos.
    """
    hoy = datetime.datetime.now()
    fecha_fin = ee.Date.fromYMD(hoy.year, hoy.month, 1)
    fecha_inicio = fecha_fin.advance(-3, "month")

    lluvia_3m_img = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(fecha_inicio, fecha_fin)
        .select("precipitation")
        .sum()
    )

    lluvia_3m = lluvia_3m_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=ESCALA_CHIRPS,
        maxPixels=1e9,
    ).get("precipitation").getInfo()

    mes_ref_fecha = datetime.datetime(hoy.year, hoy.month, 1) - pd.DateOffset(months=1)
    mes_ref = int(mes_ref_fecha.month)

    params = parametros_gamma.get(mes_ref)
    if params is None or lluvia_3m is None:
        return None, lluvia_3m, mes_ref, fecha_inicio.format("YYYY-MM-dd").getInfo(), fecha_fin.format("YYYY-MM-dd").getInfo()

    cdf_gamma = st.gamma.cdf(
        lluvia_3m,
        params["shape"],
        loc=params["loc"],
        scale=params["scale"],
    )
    cdf_ajustada = params["prob_cero"] + (1 - params["prob_cero"]) * cdf_gamma
    cdf_ajustada = np.clip(cdf_ajustada, 0.0001, 0.9999)
    spi3 = st.norm.ppf(cdf_ajustada)

    return spi3, lluvia_3m, mes_ref, fecha_inicio.format("YYYY-MM-dd").getInfo(), fecha_fin.format("YYYY-MM-dd").getInfo()


def clasificar_spi3(spi3):
    if spi3 is None:
        return 0, "Sin datos SPI-3"
    if spi3 <= UMBRALES["SPI3"]["emergencia"]:
        return 4, "Emergencia climatica"
    if spi3 <= UMBRALES["SPI3"]["alerta"]:
        return 3, "Alerta climatica"
    if spi3 <= UMBRALES["SPI3"]["prealerta"]:
        return 2, "Prealerta climatica"
    if spi3 <= UMBRALES["SPI3"]["vigilancia"]:
        return 1, "Vigilancia climatica"
    return 0, "Condicion climatica normal"


def obtener_mes_modis_disponible(geom, max_retroceso=6):
    """
    Busca el ultimo mes cerrado con imagenes MODIS disponibles.
    """
    hoy = datetime.datetime.now()

    for i in range(1, max_retroceso + 1):
        fecha_ref = datetime.datetime(hoy.year, hoy.month, 1) - pd.DateOffset(months=i)
        year = int(fecha_ref.year)
        month = int(fecha_ref.month)

        inicio = ee.Date.fromYMD(year, month, 1)
        fin = inicio.advance(1, "month")

        col = (
            ee.ImageCollection("MODIS/061/MOD13Q1")
            .filterDate(inicio, fin)
            .filterBounds(geom)
            .select("NDVI")
        )

        if col.size().getInfo() > 0:
            return year, month

    return None, None


def calcular_vci_mes(geom, year, month):
    """
    Calcula el VCI para un año y mes especificos.
    """
    if year is None or month is None:
        return None

    inicio = ee.Date.fromYMD(year, month, 1)
    fin = inicio.advance(1, "month")

    col_actual = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterDate(inicio, fin)
        .filterBounds(geom)
        .select("NDVI")
    )

    if col_actual.size().getInfo() == 0:
        return None

    ndvi_actual = col_actual.median().multiply(0.0001).rename("NDVI_actual").clip(geom)

    ndvi_hist_mes = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterBounds(geom)
        .filter(ee.Filter.calendarRange(2000, 2023, "year"))
        .filter(ee.Filter.calendarRange(month, month, "month"))
        .select("NDVI")
        .map(lambda img: img.multiply(0.0001).copyProperties(img, ["system:time_start"]))
    )

    if ndvi_hist_mes.size().getInfo() == 0:
        return None

    ndvi_min = ndvi_hist_mes.min().rename("NDVI_min").clip(geom)
    ndvi_max = ndvi_hist_mes.max().rename("NDVI_max").clip(geom)

    denominador = ndvi_max.subtract(ndvi_min)
    denominador = denominador.where(denominador.eq(0), 0.0001)

    vci = (
        ndvi_actual.subtract(ndvi_min)
        .divide(denominador)
        .multiply(100)
        .clamp(0, 100)
        .rename("VCI")
        .clip(geom)
    )

    return vci


def promedio_imagen(img, geom, banda, escala):
    if img is None:
        return None
    try:
        valor = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=escala,
            maxPixels=1e9,
        ).get(banda).getInfo()
        return valor
    except Exception:
        return None


def clasificar_vci(vci):
    if vci is None:
        return 0, "Sin datos VCI"
    if vci < UMBRALES["VCI"]["emergencia"]:
        return 4, "Emergencia vegetativa"
    if vci < UMBRALES["VCI"]["alerta"]:
        return 3, "Alerta vegetativa"
    if vci < UMBRALES["VCI"]["prealerta"]:
        return 2, "Prealerta vegetativa"
    if vci < UMBRALES["VCI"]["vigilancia"]:
        return 1, "Vigilancia vegetativa"
    return 0, "Condicion vegetativa normal"


def clasificar_vci_imagen(vci, geom):
    if vci is None:
        return ee.Image(0).rename("VCI_clase").clip(geom)

    return (
        ee.Image(0)
        .where(vci.lte(50).And(vci.gt(40)), 1)
        .where(vci.lte(40).And(vci.gt(30)), 2)
        .where(vci.lte(30).And(vci.gt(20)), 3)
        .where(vci.lte(20), 4)
        .rename("VCI_clase")
        .clip(geom)
    )


def normalizar_imagen(img, geom, nombre_banda, escala):
    stats = img.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=geom,
        scale=escala,
        maxPixels=1e9,
    )
    min_val = ee.Number(stats.get(f"{nombre_banda}_min"))
    max_val = ee.Number(stats.get(f"{nombre_banda}_max"))
    denominador = max_val.subtract(min_val).max(0.0001)
    return img.subtract(min_val).divide(denominador).clamp(0, 1)


def calcular_iiss(geom, pendiente):
    """
    Calcula IISS usando NDVI P10 y pendiente.
    """
    modis_ndvi_hist = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterDate("2000-01-01", "2023-12-31")
        .filterBounds(geom)
        .select("NDVI")
        .map(lambda img: img.multiply(0.0001).copyProperties(img, ["system:time_start"]))
    )

    ndvi_p10 = modis_ndvi_hist.reduce(ee.Reducer.percentile([10])).rename("NDVI_P10").clip(geom)

    ndvi_norm = normalizar_imagen(ndvi_p10, geom, "NDVI_P10", ESCALA_MODIS)
    vuln_vegetacion = ee.Image(1).subtract(ndvi_norm).rename("Vuln_Vegetacion")

    pendiente_norm = normalizar_imagen(pendiente, geom, "Slope", ESCALA_DEM).rename("Vuln_Pendiente")

    iiss = (
        vuln_vegetacion.multiply(0.70)
        .add(pendiente_norm.multiply(0.30))
        .rename("IISS")
        .clip(geom)
    )

    return iiss, ndvi_p10


def clasificar_iiss(iiss, geom):
    return (
        ee.Image(0)
        .where(iiss.gt(0).And(iiss.lte(0.40)), 1)
        .where(iiss.gt(0.40).And(iiss.lte(0.60)), 2)
        .where(iiss.gt(0.60).And(iiss.lte(0.80)), 3)
        .where(iiss.gt(0.80), 4)
        .rename("IISS_clase")
        .clip(geom)
    )


def crear_alerta_integrada(nivel_spi, vci_clase, iiss_clase, geom):
    """
    Integra SPI-3, VCI e IISS en una capa categorical de alerta.
    """
    spi_img = ee.Image.constant(nivel_spi).rename("SPI_clase")

    condicion_vigilancia = spi_img.gte(1).Or(vci_clase.gte(1))
    condicion_prealerta = spi_img.gte(2).And(vci_clase.gte(2)).And(iiss_clase.gte(3))
    condicion_alerta = spi_img.gte(3).And(vci_clase.gte(3)).And(iiss_clase.gte(3))
    condicion_emergencia = spi_img.gte(4).And(vci_clase.gte(4)).And(iiss_clase.eq(4))

    alerta = (
        ee.Image(0)
        .where(condicion_vigilancia, 1)
        .where(condicion_prealerta, 2)
        .where(condicion_alerta, 3)
        .where(condicion_emergencia, 4)
        .rename("Alerta_Sequia")
        .clip(geom)
    )

    return alerta


def area_por_clase(imagen_clase, geom, escala=250):
    """
    Calcula area por clase en hectareas.
    """
    area_img = ee.Image.pixelArea().addBands(imagen_clase)

    stats = area_img.reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName="clase"),
        geometry=geom,
        scale=escala,
        maxPixels=1e9,
    )

    grupos = stats.getInfo().get("groups", [])
    registros = []
    for g in grupos:
        registros.append({"clase": int(g["clase"]), "area_ha": round(g["sum"] / 10000, 2)})

    return pd.DataFrame(registros)


def agregar_leyenda_alerta(Map):
    """Agrega leyenda con compatibilidad para versiones distintas de geemap."""
    labels = [NOMBRES_ALERTA[i] for i in range(0, 5)]
    colors = [COLORES_ALERTA[i] for i in range(0, 5)]
    try:
        Map.add_legend(title="Nivel de alerta", keys=labels, colors=colors)
    except TypeError:
        Map.add_legend(title="Nivel de alerta", labels=labels, colors=colors)
    return Map


def construir_grafico_spi(df_spi):
    df_plot = df_spi.dropna(subset=["SPI3"]).copy()
    df_plot["color"] = df_plot["SPI3"].apply(lambda x: "#d73027" if x < 0 else "#4575b4")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(df_plot["date"], df_plot["SPI3"], width=25, color=df_plot["color"], alpha=0.75)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(-1.0, color="orange", linestyle="--", label="SPI <= -1.0")
    ax.axhline(-1.5, color="red", linestyle="--", label="SPI <= -1.5")
    ax.axhline(-2.0, color="darkred", linestyle="--", label="SPI <= -2.0")
    ax.set_title("SPI-3 historico - Microcuenca Amatitan")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("SPI-3")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


# ==============================================================================
# INTERFAZ STREAMLIT
# ==============================================================================

st.title("🌱 SAT de Sequía Agrícola - Microcuenca Amatitán")
st.caption("Geoportal operativo con Google Earth Engine, CHIRPS, MODIS, FABDEM, SPI-3, VCI e IISS")

with st.sidebar:
    st.header("Configuración")
    project_id = st.text_input("Proyecto Google Earth Engine", value=DEFAULT_PROJECT_ID)

    st.subheader("Línea base CHIRPS")
    anio_inicio = st.number_input("Año inicial", min_value=1981, max_value=2023, value=1981, step=1)
    anio_fin = st.number_input("Año final", min_value=1990, max_value=2025, value=2023, step=1)

    st.subheader("Datos MODIS")
    max_retroceso_modis = st.slider("Meses de retroceso para buscar MODIS", 1, 12, 6)

    st.subheader("Capas del mapa")
    mostrar_alerta = st.checkbox("Alerta integrada", value=True)
    mostrar_iiss = st.checkbox("IISS", value=True)
    mostrar_vci = st.checkbox("VCI actual", value=False)
    mostrar_drenaje = st.checkbox("Red de drenaje", value=True)

    st.markdown("---")
    if st.button("Actualizar cálculos / limpiar caché"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


ok, msg = inicializar_gee(project_id)
if not ok:
    st.error(msg)
    st.info("Si estás en Streamlit Cloud, configura una cuenta de servicio en .streamlit/secrets.toml. Si estás local, ejecuta `earthengine authenticate` en la terminal.")
    st.stop()
else:
    st.sidebar.success("GEE conectado")

try:
    microcuenca, geom, drenaje, dem = cargar_assets()
except Exception as e:
    st.error("No se pudieron cargar los assets de Earth Engine. Revisa las rutas de microcuenca, drenaje y DEM.")
    st.exception(e)
    st.stop()


# ==============================================================================
# CALCULOS PRINCIPALES
# ==============================================================================

with st.spinner("Calculando indicadores del SAT..."):
    try:
        area_km2 = geom.area().divide(1e6).getInfo()

        pendiente = calcular_pendiente(dem, geom)

        df_lluvia = construir_serie_chirps_cached(project_id, int(anio_inicio), int(anio_fin))
        df_spi, parametros_gamma = calcular_spi3_gamma(df_lluvia)
        spi3_actual, lluvia_3m_actual, mes_ref_spi, fecha_inicio_spi, fecha_fin_spi = calcular_spi3_actual(geom, parametros_gamma)
        nivel_spi, texto_spi = clasificar_spi3(spi3_actual)

        anio_ref_vci, mes_ref_vci = obtener_mes_modis_disponible(geom, max_retroceso=max_retroceso_modis)
        vci_actual = calcular_vci_mes(geom, anio_ref_vci, mes_ref_vci)
        vci_promedio = promedio_imagen(vci_actual, geom, "VCI", ESCALA_MODIS)
        nivel_vci, texto_vci = clasificar_vci(vci_promedio)
        vci_clase = clasificar_vci_imagen(vci_actual, geom)

        iiss, ndvi_p10 = calcular_iiss(geom, pendiente)
        iiss_clase = clasificar_iiss(iiss, geom)

        alerta_integrada = crear_alerta_integrada(nivel_spi, vci_clase, iiss_clase, geom)
        df_area_alerta = area_por_clase(alerta_integrada, geom, escala=ESCALA_MODIS)

    except Exception as e:
        st.error("Ocurrió un error durante el cálculo del SAT.")
        st.exception(e)
        st.stop()


# ==============================================================================
# PESTAÑAS
# ==============================================================================

tab_monitoreo, tab_mapa, tab_spi, tab_areas, tab_descargas, tab_metodo = st.tabs(
    ["Monitoreo actual", "Mapa", "SPI histórico", "Áreas", "Descargas", "Metodología"]
)


# ------------------------------------------------------------------------------
# TAB 1: MONITOREO ACTUAL
# ------------------------------------------------------------------------------

with tab_monitoreo:
    st.subheader("Indicadores actuales del SAT")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Área microcuenca", f"{area_km2:.2f} km²")

    with col2:
        if spi3_actual is not None:
            st.metric("SPI-3 actual", f"{spi3_actual:.2f}", texto_spi)
        else:
            st.metric("SPI-3 actual", "Sin datos", texto_spi)

    with col3:
        if lluvia_3m_actual is not None:
            st.metric("Lluvia 3 meses", f"{lluvia_3m_actual:.1f} mm")
        else:
            st.metric("Lluvia 3 meses", "Sin datos")

    with col4:
        if vci_promedio is not None:
            st.metric("VCI promedio", f"{vci_promedio:.1f}", texto_vci)
        else:
            st.metric("VCI promedio", "Sin datos", texto_vci)

    with col5:
        estado_general = max(nivel_spi, nivel_vci)
        st.metric("Estado general", NOMBRES_ALERTA.get(estado_general, "Sin datos"))

    st.markdown("---")
    st.subheader("Interpretación operativa")

    if nivel_spi >= 4 and nivel_vci >= 4:
        st.error("EMERGENCIA: activar respuesta prioritaria en zonas IISS Muy Alta.")
    elif nivel_spi >= 3 and nivel_vci >= 3:
        st.warning("ALERTA: activar medidas de mitigación en zonas IISS Alta y Muy Alta.")
    elif nivel_spi >= 2 and nivel_vci >= 2:
        st.warning("PREALERTA: iniciar monitoreo quincenal y comunicación preventiva con productores.")
    elif nivel_spi >= 1 or nivel_vci >= 1:
        st.info("VIGILANCIA: mantener monitoreo mensual y revisar pronóstico climático.")
    else:
        st.success("NORMAL: continuar monitoreo rutinario.")

    st.markdown("#### Periodos usados")
    colp1, colp2 = st.columns(2)
    with colp1:
        st.write(f"**Periodo SPI-3:** {fecha_inicio_spi} a {fecha_fin_spi}")
        st.write(f"**Mes de referencia SPI:** {mes_ref_spi}")
    with colp2:
        if anio_ref_vci is not None:
            st.write(f"**Mes VCI disponible:** {anio_ref_vci}-{mes_ref_vci:02d}")
        else:
            st.write("**Mes VCI disponible:** sin datos")


# ------------------------------------------------------------------------------
# TAB 2: MAPA
# ------------------------------------------------------------------------------

with tab_mapa:
    st.subheader("Mapa integrado de alerta de sequía agrícola")

    vis_alerta = {
        "min": 0,
        "max": 4,
        "palette": [COLORES_ALERTA[i] for i in range(0, 5)],
    }
    vis_iiss = {
        "min": 0,
        "max": 1,
        "palette": ["#ffffcc", "#fed976", "#fd8d3c", "#fc4e2a", "#bd0026", "#800026"],
    }
    vis_vci = {
        "min": 0,
        "max": 100,
        "palette": ["red", "yellow", "green"],
    }

    Map = geemap.Map()
    Map.centerObject(microcuenca, 13)
    Map.add_basemap("SATELLITE")

    if mostrar_alerta:
        Map.addLayer(alerta_integrada, vis_alerta, "Alerta integrada")
    if mostrar_iiss:
        Map.addLayer(iiss, vis_iiss, "IISS")
    if mostrar_vci and vci_actual is not None:
        Map.addLayer(vci_actual, vis_vci, "VCI actual")
    if mostrar_drenaje:
        Map.addLayer(drenaje, {"color": "00ffff"}, "Red de drenaje")

    Map.addLayer(microcuenca, {"color": "white", "fillColor": "00000000"}, "Microcuenca")
    agregar_leyenda_alerta(Map)

    st_folium(Map, width=None, height=650)

    st.caption("La alerta integrada combina SPI-3, VCI e IISS. El IISS indica susceptibilidad espacial, no sequía actual por sí solo.")


# ------------------------------------------------------------------------------
# TAB 3: SPI HISTORICO
# ------------------------------------------------------------------------------

with tab_spi:
    st.subheader("Serie histórica SPI-3")
    fig_spi = construir_grafico_spi(df_spi)
    st.pyplot(fig_spi)

    with st.expander("Ver tabla SPI-3"):
        st.dataframe(df_spi[["date", "rainfall", "rain_3m", "SPI3"]], use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 4: AREAS
# ------------------------------------------------------------------------------

with tab_areas:
    st.subheader("Área por nivel de alerta")

    if not df_area_alerta.empty:
        df_area_alerta = df_area_alerta.copy()
        df_area_alerta["nivel"] = df_area_alerta["clase"].map(NOMBRES_ALERTA)
        df_area_alerta = df_area_alerta.sort_values("clase")

        st.dataframe(df_area_alerta[["clase", "nivel", "area_ha"]], use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(df_area_alerta["nivel"], df_area_alerta["area_ha"], color=[COLORES_ALERTA.get(c, "gray") for c in df_area_alerta["clase"]])
        ax.set_title("Área por nivel de alerta")
        ax.set_xlabel("Nivel de alerta")
        ax.set_ylabel("Área ha")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No se pudo calcular el área por nivel de alerta.")


# ------------------------------------------------------------------------------
# TAB 5: DESCARGAS
# ------------------------------------------------------------------------------

with tab_descargas:
    st.subheader("Descarga de resultados")

    csv_spi = df_spi.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar serie SPI-3 CSV",
        data=csv_spi,
        file_name="spi3_amatitan.csv",
        mime="text/csv",
    )

    if not df_area_alerta.empty:
        csv_area = df_area_alerta.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Descargar áreas por alerta CSV",
            data=csv_area,
            file_name="areas_alerta_amatitan.csv",
            mime="text/csv",
        )


# ------------------------------------------------------------------------------
# TAB 6: METODOLOGIA
# ------------------------------------------------------------------------------

with tab_metodo:
    st.subheader("Metodología del SAT")
    st.markdown(
        """
        Este geoportal implementa un prototipo operativo de Sistema de Alerta Temprana de sequía agrícola para la microcuenca Amatitán.

        **Componentes principales:**

        1. **SPI-3:** indicador climático basado en la precipitación acumulada de tres meses de CHIRPS.
        2. **VCI:** indicador de condición de la vegetación calculado con MODIS NDVI.
        3. **IISS:** índice espacial de susceptibilidad construido con NDVI P10 histórico y pendiente derivada del DEM FABDEM.
        4. **Alerta integrada:** combinación de déficit climático, estrés vegetal y susceptibilidad territorial.

        **Lectura operativa:**

        - SPI-3 indica **cuándo** existe déficit de lluvia.
        - VCI indica **si la vegetación ya muestra estrés**.
        - IISS indica **dónde priorizar la intervención**.

        **Advertencia técnica:**
        Este sistema debe validarse con datos locales de campo, reportes agrícolas, fechas de siembra, rendimientos y registros comunitarios de afectación.
        """
    )

st.markdown("---")
st.caption("MIALEMPA agente IA CRL sobre la Cuenca del río Lempa, El Salvador.")
