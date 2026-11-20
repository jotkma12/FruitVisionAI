# FruitVision AI

FruitVision AI clasifica imágenes de frutas y verduras con un modelo EfficientNetB0 ya entrenado. La persona selecciona una fotografía en una interfaz Streamlit; el frontend envía sus bytes al servicio FastAPI, que prepara la imagen y consulta el modelo. La aplicación devuelve la categoría, el estado (fresco o podrido) y la puntuación asociada a la predicción.

El proyecto contiene la aplicación de inferencia y sus pruebas. El entrenamiento y la evaluación ya realizados se conservan como entregables; la aplicación no vuelve a entrenar el modelo.

## Demo en línea

Prueba FruitVision AI directamente desde el repositorio, sin instalar nada.

### Aplicación web

[**Abrir FruitVision AI**](https://fruitvision-ai-frontend.onrender.com)

Desde la aplicación puedes cargar una imagen de una fruta o verdura y obtener una predicción real del modelo: si está fresca o podrida, la clase identificada y el nivel de confianza.

### API del modelo

- [Backend](https://fruitvision-ai-backend.onrender.com)
- [Estado del servicio](https://fruitvision-ai-backend.onrender.com/health)
- [Información del modelo y sus clases](https://fruitvision-ai-backend.onrender.com/model/info)
- [Documentación interactiva de la API](https://fruitvision-ai-backend.onrender.com/docs)

### Instrucciones para probar

1. Abre el [enlace de la aplicación web](https://fruitvision-ai-frontend.onrender.com).
2. Espera a que cargue.
3. Selecciona una imagen de una fruta o verdura.
4. Ejecuta la predicción.
5. Consulta el resultado generado por el modelo.

> Los servicios gratuitos de Render pueden suspenderse por inactividad. El primer acceso puede tardar algunos minutos en responder.

## Funcionalidades

- Seleccionar y previsualizar una fotografía JPEG, PNG o WEBP.
- Validar la imagen y comunicar el progreso, los errores de conexión o los fallos de inferencia.
- Consultar el estado del servicio y la información del modelo.
- Ejecutar una predicción con el modelo original y mostrar la categoría, el estado y la puntuación devuelta.
- Conservar únicamente el resultado correspondiente a la imagen seleccionada actualmente.

El clasificador contempla las 20 categorías del archivo `models/class_names.json`: diez categorías Fresh y diez Rotten, para manzana, banano, pimentón, zanahoria, pepino, mango, naranja, papa, fresa y tomate.

## Tecnologías

| Tecnología | Uso |
|---|---|
| Python 3.12 | Lenguaje de backend, cliente e interfaz |
| TensorFlow 2.19.1 y Keras 3.15.1 | Carga e inferencia con el modelo `.keras` |
| FastAPI y Uvicorn | API HTTP y servidor ASGI |
| Streamlit | Interfaz de carga, vista previa y resultados |
| Pillow | Decodificación, conversión RGB y redimensionamiento |
| Requests | Comunicación HTTP entre la interfaz y el backend |
| pytest y HTTPX | Pruebas de componentes, cliente y API |

Las versiones exactas de las dependencias están declaradas en [requirements.txt](requirements.txt).

## Estructura del proyecto

```text
FruitVisionAI/
  README.md
  requirements.txt
  models/                         # modelo oficial y etiquetas
  models/training/                # modelos del entrenamiento
  notebooks/fruitvision_ai_entrenamiento.ipynb
  data/processed/                 # train, validation, test, manifest_clean
  reports/                        # metricas, informes, curvas e historial
  src/backend/
  src/frontend/
  tests/
  docs/
```

El notebook y los artefactos de entrenamiento y evaluacion se conservan en `notebooks/`, `data/processed/`, `models/training/` y `reports/`. La configuracion predeterminada de la aplicacion carga el modelo oficial `models/fruitvision_ai_model.keras` y las etiquetas `models/class_names.json`; no selecciona modelos de entrenamiento.

## Entregables de entrenamiento y evaluacion

- `notebooks/fruitvision_ai_entrenamiento.ipynb`
- `data/processed/train.csv`, `data/processed/validation.csv`, `data/processed/test.csv` y `data/processed/manifest_clean.csv`
- `models/training/fruitvision_ai_model.keras`, `models/training/mejor_modelo_inicial.keras` y `models/training/mejor_modelo_finetuning.keras`
- `reports/classification_report.csv`, `reports/confusion_matrix.csv`, `reports/errores_confusion.csv` y `reports/conflictos_excluidos.csv`
- `reports/training_history.csv`, `reports/curvas_entrenamiento.png`, `reports/metrics.json` y `reports/training_metadata/class_names.json`

El backend toma por defecto el modelo de produccion de `models/fruitvision_ai_model.keras` y las etiquetas de `models/class_names.json`. Las variables de entorno de configuracion documentadas mas abajo permiten adaptar esas rutas en otros entornos.

## Requisitos previos

- Windows de 64 bits.
- Python 3.12.
- Acceso a los archivos `models/fruitvision_ai_model.keras` y `models/class_names.json`.

Ejecuta los siguientes comandos en PowerShell desde la raíz del proyecto. Si ya existe `.venv`, conserva ese entorno y omite su creación.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

## Iniciar la aplicación

Abre dos terminales de PowerShell en la raíz del proyecto. En la primera, inicia el backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.backend.api:app --host 127.0.0.1 --port 8000
```

En la segunda, inicia Streamlit:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/frontend/app.py
```

Abre la dirección local que Streamlit muestra en la terminal (por defecto, [http://localhost:8501](http://localhost:8501)). La API escucha por defecto en [http://127.0.0.1:8000](http://127.0.0.1:8000).

FastAPI ofrece su documentación interactiva en [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) y su esquema OpenAPI en `/openapi.json`.

## Endpoints

| Método y ruta | Propósito |
|---|---|
| `GET /health` | Informa si el modelo está listo para inferencia. |
| `GET /model/info` | Devuelve el nombre del modelo y las 20 categorías en el orden cargado. |
| `POST /predict` | Recibe el archivo en el campo multipart `image` y devuelve categoría, estado y puntuación. |

La carga admite JPEG, PNG y WEBP hasta el límite predeterminado de 10 MiB. El servicio responde con error para archivos vacíos, corruptos, no admitidos o superiores al límite; si el modelo no está disponible, no genera una predicción sustituta.

## Modelos y configuración

Las rutas predeterminadas son:

- `models/fruitvision_ai_model.keras`
- `models/class_names.json`

La aplicación carga las categorías desde el JSON y comprueba que su cantidad coincida con la salida del modelo. La configuración puede sobrescribirse mediante estas variables de entorno:

| Variable | Valor predeterminado | Función |
|---|---|---|
| `FRUITVISION_MODEL_PATH` | `models/fruitvision_ai_model.keras` | Ruta del modelo Keras |
| `FRUITVISION_CLASS_NAMES_PATH` | `models/class_names.json` | Ruta de las categorías |
| `FRUITVISION_MAX_UPLOAD_BYTES` | `10485760` | Tamaño máximo de carga en bytes |
| `FRUITVISION_API_URL` | `http://127.0.0.1:8000` | URL base del backend usada por el frontend |

## Ejecutar las pruebas

Desde la raíz y con las dependencias instaladas:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

La prueba integrada con el modelo real puede ejecutarse por separado:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_integration_model.py -q
```

La ejecución completa registrada el 19 de noviembre de 2026 obtuvo 20 pruebas aprobadas en 8.65 segundos y una advertencia de obsolescencia de Starlette relacionada con el uso de HTTPX en `TestClient`. La advertencia no hizo fallar pruebas. El detalle de la verificación y la evidencia del preprocesamiento están en el [documento técnico](docs/technical.md).

## Documentación relacionada

- [Documento técnico académico](docs/technical.md)
- [Guía de demostración](docs/demo.md)
- [Inventario de integridad de artefactos](docs/artifact-integrity.md)

El [notebook de entrenamiento](notebooks/fruitvision_ai_entrenamiento.ipynb) se conserva como entregable existente. La aplicación no lo ejecuta ni lo modifica.
