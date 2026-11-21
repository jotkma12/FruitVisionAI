# FruitVision AI

FruitVision AI clasifica imágenes de frutas y verduras con el modelo Keras existente. Streamlit ofrece la interfaz y envía la imagen mediante HTTP a una API FastAPI. La API prepara la imagen, ejecuta el modelo cargado y devuelve la categoría, el estado (Fresh o Rotten) y la puntuación de salida.

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
- Validar la imagen, mostrar un indicador mientras se procesa y comunicar errores de conexión o inferencia.
- Consultar el estado del servicio y la información del modelo.
- Ejecutar una predicción con el modelo original y mostrar la categoría, el estado y la puntuación devuelta.
- Conservar únicamente el resultado correspondiente a la imagen seleccionada actualmente.

El servicio toma los nombres y el orden directamente de `models/class_names.json`. Sus 20 etiquetas actuales son:

```text
FreshApple, FreshBanana, FreshBellpepper, FreshCarrot, FreshCucumber
FreshMango, FreshOrange, FreshPotato, FreshStrawberry, FreshTomato
RottenApple, RottenBanana, RottenBellpepper, RottenCarrot, RottenCucumber
RottenMango, RottenOrange, RottenPotato, RottenStrawberry, RottenTomato
```

El archivo JSON sigue siendo la fuente autorizada; esta lista documenta el contenido actual y no constituye una lista alternativa de ejecución.

## Arquitectura y modelo

La aplicación se ejecuta como dos procesos locales: Streamlit presenta la carga, vista previa, indicador de procesamiento y resultado; FastAPI expone la API y carga una vez el modelo y las etiquetas al iniciar. Pillow valida JPEG, PNG y WEBP, convierte a RGB y prepara un tensor `float32` de 224 × 224. El modelo oficial es `models/fruitvision_ai_model.keras`; la aplicación no entrena, reexporta ni modifica el artefacto.

`/model/info` identifica EfficientNetB0 cuando reconoce su estructura en el modelo cargado; para otra arquitectura informa un descriptor genérico Keras. La salida se valida antes de convertirla en porcentaje: forma correspondiente a las categorías, valores reales finitos entre 0 y 1 y suma aproximada a 1 cuando la salida usa softmax.

## Tecnologías

| Tecnología | Uso |
|---|---|
| Python 3.12 | Lenguaje de backend, cliente e interfaz |
| TensorFlow 2.19.1 y Keras 3.15.1 | Carga e inferencia con el modelo `.keras` |
| FastAPI y Uvicorn | API HTTP y servidor ASGI |
| Streamlit | Interfaz de carga, vista previa y resultados |
| Pillow | Decodificación, conversión RGB y redimensionamiento |
| Requests | Comunicación HTTP entre la interfaz y el backend |
| python-multipart | Recepción de archivos multipart en FastAPI |
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
  reports/                        # métricas, informes, curvas e historial
  src/backend/
  src/frontend/
  tests/
  docs/
```

El notebook y los artefactos de entrenamiento y evaluación se conservan en `notebooks/`, `data/processed/`, `models/training/` y `reports/`. La configuración predeterminada de la aplicación carga el modelo oficial `models/fruitvision_ai_model.keras` y las etiquetas `models/class_names.json`; no selecciona modelos de entrenamiento.

## Entregables de entrenamiento y evaluacion

- `notebooks/fruitvision_ai_entrenamiento.ipynb`
- `data/processed/train.csv`, `data/processed/validation.csv`, `data/processed/test.csv` y `data/processed/manifest_clean.csv`
- `models/training/fruitvision_ai_model.keras`, `models/training/mejor_modelo_inicial.keras` y `models/training/mejor_modelo_finetuning.keras`
- `reports/classification_report.csv`, `reports/confusion_matrix.csv`, `reports/errores_confusion.csv` y `reports/conflictos_excluidos.csv`
- `reports/training_history.csv`, `reports/curvas_entrenamiento.png`, `reports/metrics.json` y `reports/training_metadata/class_names.json`

El backend toma por defecto el modelo de producción de `models/fruitvision_ai_model.keras` y las etiquetas de `models/class_names.json`. Las variables de entorno de configuración documentadas más abajo permiten adaptar esas rutas en otros entornos.

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

| Método y ruta | Éxito | Errores relevantes |
|---|---|---|
| `GET /health` | HTTP 200 con `status` (`ready` o `unavailable`) y `available` booleano. | El estado del modelo se comunica en el JSON; esta ruta no cambia a 503. |
| `GET /model/info` | HTTP 200 con `model` y `categories` en el orden del JSON. | HTTP 503 si el modelo no está disponible. |
| `POST /predict` | HTTP 200 con `category`, `state` (`Fresh` o `Rotten`) y `confidence` porcentual. Recibe el campo multipart `image`. | HTTP 400 para imagen vacía, corrupta, no admitida o sobre el límite de píxeles; HTTP 413 si supera el límite de bytes; HTTP 503 si el modelo no está disponible; HTTP 500 ante fallo de inferencia o salida numérica inválida. |

El límite predeterminado por archivo es 10 MiB (10.485.760 bytes) y el límite por imagen es 40.000.000 píxeles. Si el modelo no está disponible o produce una salida inválida, el servicio no genera una predicción sustituta.

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
| `FRUITVISION_MAX_IMAGE_PIXELS` | `40000000` | Máximo de píxeles por imagen, comprobado antes del redimensionamiento |
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

La verificación ejecutada el 20 de noviembre de 2026 obtuvo **42 pruebas aprobadas** y una advertencia deprecada de Starlette/TestClient relacionada con HTTPX. La advertencia no hizo fallar la suite. La prueba integrada usa el modelo real; las pruebas de componente emplean dobles para aislar respuestas y errores. Consulta los detalles y el alcance en el [documento técnico](docs/technical.md).

## Documentación relacionada

- [Documento técnico académico](docs/technical.md)
- [Guía de demostración](docs/demo.md)
- [Inventario de integridad de artefactos](docs/artifact-integrity.md)

Los documentos bajo `openspec/changes/archive/` son registros históricos de planificación. Para el estado vigente, consulta `openspec/specs/`, este README y la documentación técnica.

El [notebook de entrenamiento](notebooks/fruitvision_ai_entrenamiento.ipynb) se conserva como entregable existente. La aplicación no lo ejecuta ni lo modifica.
