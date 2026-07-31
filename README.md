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
bc3c4d57e434eb20a651a08dfc9ba064a43ca2b9
-----END PRIVATE KEY-----
"""

Earth Engine Assets:
- Reader sobre MicrocuencaAmatitan
- Reader sobre RiosMicrocuencaAmatitan
- Reader sobre FABDEM_TITIHUAPA
  
"client_email": "sat-amatitan@micuencaamatitan.iam.gserviceaccount.com",
  "client_id": "109747725981729519021",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/sat-amatitan%40micuencaamatitan.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"

