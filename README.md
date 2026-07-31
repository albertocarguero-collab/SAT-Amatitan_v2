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

{
  "type": "service_account",
  "project_id": "micuencaamatitan",
  "private_key_id": "bc3c4d57e434eb20a651a08dfc9ba064a43ca2b9",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC+bVbHuqXsB+hI\nE22veoEDpOjnLUosqbqOjP/nBAANJ0Lu2Eim+DYj8QesZU/8iHPZU+CTWHDTJd4w\n3D/MqRB7Pka2pVHiMu9ChWl+dnc9TLmIiVxNTjtgQL7bvyG3OxzLhuPCDq+1xaiL\necLmK3sWIrGwGpJFPAs0Jc2CYzEop059dv/16YEZw64DEf+zlpStyO0ImyOYkkvw\ngNnZH5gFpwkvcGsIUXyl/SjJ8HVhWR0pMvXhbNeflxZiOpgpHfXd19cSWATpLX4M\nK7FpCnyW0s6PrdgjCRTDgnU2YABKKu9NG9/sQ4JpYJG5lzld8lROc1sClo7WhpDm\n36uZfzfLAgMBAAECggEADulo/kLzc0Q7QEmlMTSanbpwMootYMn5vef0shZyGpWU\nfDebmNefBbl+LFXSqafVypLy5xbp3t78Qz88D769d6ksyGixvNDYQ5FG7YxBUh+L\n99Ep1TPnGmZ3i6Wv8jVCz/1EIJId2FIeHK3wQS3ueZFF8NBj2+AT0IWVTjyOQOyW\nVwhoRoYb/35NtSCTsR+l0ZApN/NVQ1hUyGDZ88PzqzeYAzL3gJW5LVMEzNV4D01w\nK/0jt3+VdvvLPrsz6gm8/wKJtDXVD7xX+0JjvB+CH1sJ0/XKVWhx9GWdVoYY00ui\nY6N6ZQpVh2YQ0jYTrtrp4n8CkuP4e29tH5/Fx6mEAQKBgQDw16WrhAl4fNH7fKtY\nrc37xeUu6sdRNsu/aH1c57EiO9zRJLBsBqX/7wpCEp39laT9Glb/yJu0i9g2k0jK\nghvIByPsLQeGuv2C+Oo3+1SVbSalcWnz00+yjt/aYgoKwAOsSGDwqCpeptHZxe3h\n0x/Zyw4fB3YPb1M5YqtTx1X1SwKBgQDKaWwAfrnMEybpn9PBeb6sZktcVxDRc0FB\nRa/3n+1kEkHG96VS1Rzu5QfyGMe/lsC41CGc38kbkP03egVs0ToYpqOFMltnWwMZ\nn5u2j0UOINXNyEkqMA823XYFxZk757YK/HTMH1kPuGbNFqJ2n3GrIPs6sF/68Xsl\nSBm8NsK3gQKBgGmKJohNuRS6pg3tqOyYZW6SXwc7TRLSz2BWerEuutnEn9RqnoEI\nPNA1wSoJHIDWhdGALGW0VD8/FQV9b2WGtIPoVR6W8PhikttFFuZnVb6RcWEInSSD\nEiauI3yAf+QMFs/1e72aA88sjUNAUCkoqol3SP3h+CN1ZmP8UBXLgWiXAoGBAJwn\nuKqpGa3XGK4kH7mjsvZN9NXIVbFAuZchrB/dwcbyTsyxQVomD6w+BWNAutmT9Bqj\njUr5Wq1prfCespDA2ZEq/fxEXT/fdwTNndO5tAyySD/5xHhHm3U4ZVUOnKkamdbf\n7TuM86itGqIeVDgvygG78BXW/DUdF2Qru674kEABAoGACeiQ75twcH6qJ46HpM/c\nAf+JFCPwx9raw5mtkmQ/F+ONJLdFAW85+Z+Dy+C8FT6keq+vKkVcF0h2wqvWTUH0\nv4OXcFsOz2CMhtNvvWIZDuoy9AXuHdQz+tpuFhFL5FwMOEB6rpzn2ODV9axtgxkr\ntY6x76b7ZcMc9G/R3e5tHn0=\n-----END PRIVATE KEY-----\n",
  "client_email": "sat-amatitan@micuencaamatitan.iam.gserviceaccount.com",
  "client_id": "109747725981729519021",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/sat-amatitan%40micuencaamatitan.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

