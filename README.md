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

[gee]
service_account = "sat-amatitan@micuencaamatitan.iam.gserviceaccount.com"
project = "micuencaamatitan"
private_key = """
-----BEGIN PRIVATE KEY-----
TU_LLAVE_PRIVADA
-----END PRIVATE KEY-----
"""

La cuenta de servicio debe tener permiso de lectura sobre los assets de Earth Engine.
projects/micuencaamatitan/assets/MicrocuencaAmatitan
projects/micuencaamatitan/assets/RiosMicrocuencaAmatitan
projects/micuencaamatitan/assets/FABDEM_TITIHUAPA

IAM:
- Earth Engine Resource Viewer
- Viewer opcional

Earth Engine Assets:
- Reader sobre MicrocuencaAmatitan
- Reader sobre RiosMicrocuencaAmatitan
- Reader sobre FABDEM_TITIHUAPA

