# Nora Cuántica

Demo conversacional experimental cuya predisposición inicial se deriva de una
fuente de entropía cuántica.

## Desarrollo local

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[api,dev]"
.venv\Scripts\python.exe -m uvicorn nora_quantica.api:app --app-dir src
```

La configuración se carga desde `.env`; `.env.example` documenta las variables
disponibles sin incluir secretos.

## Demo pública

La arquitectura de despliegue usa:

- Firebase Hosting como entrada HTTPS.
- Cloud Run para la aplicación FastAPI.
- Firestore para conversaciones, entropía en caché y rate limiting.
- Secret Manager para las claves de OpenAI, OpenRouter y ANU.

El contenedor se construye con `Dockerfile`. `firebase.json` dirige el tráfico
del sitio al servicio `nora-quantica` en `us-east1`.

Los secretos esperados en Secret Manager son:

- `nora-evaluator-api-key`
- `nora-generator-api-key`
- `nora-anu-qrng-api-key`
- `nora-session-secret`

La configuración no sensible de Cloud Run está en
`deploy/cloud-run.env.example.yaml`. Antes de desplegar hay que copiarla,
completar los modelos activos y mantener el archivo resultante fuera de Git.
