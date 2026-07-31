# SAT de Sequía Agrícola - Microcuenca Amatitán

Geoportal en Streamlit para monitoreo de sequía agrícola en la microcuenca del río Amatitán.

## Archivos principales

- `app.py`: aplicación principal Streamlit.
- `requirements.txt`: dependencias de Python.
- `runtime.txt`: versión de Python para Streamlit Cloud.
- `.streamlit/secrets.toml.example`: plantilla de credenciales de Google Earth Engine.

## Ejecutar localmente

pip install -r requirements.txt
earthengine authenticate
streamlit run app.py

## Streamlit Cloud

En `Manage app > Settings > Secrets`, configura:

