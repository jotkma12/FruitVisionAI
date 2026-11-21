# Documento técnico de FruitVision AI

## 1. Introducción y objetivo del sistema

FruitVision AI es un sistema local de clasificación de imágenes de frutas y verduras. Su configuración predeterminada carga el modelo Keras existente, cuya arquitectura reconocida es EfficientNetB0, y presenta la categoría, el estado del producto y la puntuación asociada a la predicción.

El alcance de este documento es la aplicación de inferencia y su verificación. El modelo, el notebook y los resultados del entrenamiento se tratan como artefactos existentes; el sistema no reentrena ni modifica esos recursos.

## 2. Requisitos funcionales y alcance

El sistema permite:

- Seleccionar una imagen admitida y ver una vista previa.
- Enviar sus bytes desde la interfaz al backend mediante HTTP.
- Validar y preparar la imagen para que coincida con la entrada esperada por el modelo.
- Consultar el estado del servicio y la información de las categorías.
- Ejecutar una inferencia y mostrar la respuesta de esa solicitud.
- Presentar errores de archivo, conexión, disponibilidad del modelo o inferencia sin sustituirlos por una predicción almacenada.

El backend carga el modelo una vez durante su inicio y lee las categorías de `models/class_names.json`. La aplicación usa los artefactos de `models/` como entradas existentes de solo lectura.

## 3. Arquitectura general

La solución separa la interfaz, el transporte HTTP, el procesamiento de imágenes y la inferencia:

1. **Frontend:** Streamlit permite seleccionar una imagen, muestra su vista previa, el progreso y el resultado.
2. **Cliente HTTP:** Requests envía los bytes al endpoint de inferencia.
3. **API:** FastAPI recibe la carga multipart y ofrece los endpoints de estado, información y predicción.
4. **Procesamiento:** Pillow valida el formato, convierte a RGB y redimensiona la imagen.
5. **Servicio de modelo:** TensorFlow/Keras carga el modelo original y calcula la predicción.
6. **Respuesta:** FastAPI devuelve categoría, estado y puntuación como JSON; Streamlit los presenta.

## 4. Diagrama de arquitectura

```mermaid
flowchart LR
    U[Usuario] --> S[Streamlit]
    S -->|HTTP multipart con la imagen| F[FastAPI]
    F --> P[Pillow: valida, convierte a RGB y redimensiona]
    P --> M[Modelo Keras cargado]
    M --> J[Respuesta JSON]
    J --> S
```

## 5. Tecnologías utilizadas y justificación

| Tecnología | Versión verificada o fijada | Responsabilidad y justificación |
|---|---:|---|
| Python | 3.12.2 verificado | Ejecuta backend, cliente e interfaz con un mismo ecosistema. |
| TensorFlow | 2.19.1 | Carga el modelo exportado y ejecuta inferencia. |
| Keras | 3.15.1 | Interfaz de carga del archivo `.keras` y ejecución del modelo. |
| FastAPI | 0.165.0 fijada | Define los endpoints HTTP y recibe archivos multipart. |
| Uvicorn | 0.54.0 fijada | Ejecuta la aplicación ASGI de FastAPI. |
| Streamlit | 1.66.1 fijada | Implementa la interfaz de carga, vista previa y resultados en Python. |
| Pillow | 11.2.1 fijada | Decodifica los formatos admitidos, convierte a RGB y redimensiona. |
| Requests | 2.34.2 fijada | Envía la imagen desde el frontend al servicio HTTP. |
| pytest | 9.2.0 fijada | Ejecuta las pruebas unitarias, de API, frontend e integración. |
| HTTPX | 0.28.1 fijada | Transporte de pruebas para el cliente ASGI usado por FastAPI. |
| python-multipart | 0.0.33 fijada | Decodifica las cargas multipart de archivos recibidas por FastAPI. |
| NumPy | 2.1.3 fijada | Representa la imagen preparada como tensor numérico. |

Las versiones fijadas se declaran en `requirements.txt`. La elección de FastAPI y Uvicorn separa la API del proceso de interfaz y facilita comprobar sus endpoints. Streamlit permite construir el flujo de usuario sin añadir una aplicación JavaScript separada. Pillow cubre las operaciones de imagen requeridas; no se necesitan transformaciones adicionales de visión por computador que justifiquen OpenCV. TensorFlow/Keras carga directamente el artefacto existente, sin convertirlo.

## 6. Descripción del modelo EfficientNetB0

La configuración predeterminada carga `models/fruitvision_ai_model.keras` con TensorFlow/Keras y obtiene nombres y orden de categorías de `models/class_names.json`. El inicio comprueba dimensiones de entrada y cantidad de salidas. El endpoint `/model/info` identifica EfficientNetB0 cuando reconoce un submodelo con ese nombre y capas estructurales indicativas; para otro modelo devuelve un descriptor genérico como `Keras Functional model`, en lugar de afirmar una arquitectura no identificada.

Las 20 etiquetas reales, en el orden de `models/class_names.json`, son `FreshApple`, `FreshBanana`, `FreshBellpepper`, `FreshCarrot`, `FreshCucumber`, `FreshMango`, `FreshOrange`, `FreshPotato`, `FreshStrawberry`, `FreshTomato`, `RottenApple`, `RottenBanana`, `RottenBellpepper`, `RottenCarrot`, `RottenCucumber`, `RottenMango`, `RottenOrange`, `RottenPotato`, `RottenStrawberry` y `RottenTomato`. El estado mostrado se deriva del prefijo de la categoría devuelta; el JSON sigue siendo la fuente de verdad en ejecución.

El SHA-256 registrado de `models/fruitvision_ai_model.keras` es `3DCAA1D4E95AE2D5EEF6FD17180BBD4A1C92506E591651AD19A0ED5D90E8E8F1`. El artefacto de entrenamiento `models/training/fruitvision_ai_model.keras` figura con el mismo hash; la configuración predeterminada del backend carga el modelo de producción de `models/`. El SHA-256 registrado de `models/class_names.json` es `91974A4A50C4F0C4E86CCBC6B5AA2444AB33ADBC2B30786374BCB7D2DC75414D`.

## 7. Evidencia del entrenamiento y evaluación

El resumen de evaluación existente se encuentra en `reports/metrics.json`. Las métricas de la sección siguiente se transcriben de ese archivo y corresponden a 1765 imágenes de prueba. No son una medición de la confianza de una predicción individual.

El notebook existente está en `notebooks/fruitvision_ai_entrenamiento.ipynb`. La evidencia de preprocesamiento se obtuvo mediante consulta de solo lectura del notebook y del modelo; no se ejecutaron celdas. El SHA-256 registrado antes y después de la consulta es `08A5BB9D2F8491FD5C9395BB3AEA5A2F3AAFF2C46AF4C721E1B2D47F0466ED12`.

## 8. Métricas verificadas

| Métrica | Resultado |
|---|---:|
| Arquitectura | EfficientNetB0 |
| Número de clases | 20 |
| Imágenes de prueba | 1.765 |
| Accuracy | 96.7705 % |
| Loss | 0.0942 |
| Precision macro | 96.6790 % |
| Recall macro | 96.6602 % |
| F1 macro | 96.6479 % |
| Precision weighted | 96.8057 % |
| Recall weighted | 96.7705 % |
| F1 weighted | 96.7668 % |
| Predicciones incorrectas | 57 |

**Fuente:** `reports/metrics.json`. Los porcentajes reproducen los valores proporcionados para esta entrega; `loss` se presenta como valor decimal.

## 9. Preprocesamiento de imágenes

La entrada compatible con el modelo tiene tamaño **224 × 224**, tres canales RGB y tipo `float32`. El backend acepta JPEG, PNG y WEBP; Pillow decodifica la imagen, la convierte a RGB y la redimensiona a ese tamaño. La matriz se entrega en rango 0–255 como `float32`.

Antes de decodificar completamente y redimensionar, el backend compara ancho × alto con el límite configurable `FRUITVISION_MAX_IMAGE_PIXELS`, cuyo valor predeterminado es **40.000.000 píxeles**. El límite de carga comprimida sigue siendo `FRUITVISION_MAX_UPLOAD_BYTES` (10 MiB por defecto). Archivos vacíos, corruptos, formatos distintos y exceso de píxeles se responden como HTTP 400; el exceso de bytes se responde como HTTP 413.

El modelo incorpora internamente las capas `Rescaling` y `Normalization`. Por tanto, la aplicación no divide los píxeles entre 255 ni aplica otra normalización externa o `preprocess_input`. Así se evita duplicar el preprocesamiento que ya forma parte del modelo.

Esta descripción coincide con la evidencia conservada en el notebook y con la entrada del modelo. El archivo `docs/artifact-integrity.md` registra las huellas SHA-256 de los artefactos verificados.

## 10. Integración entre frontend, API y modelo

Los archivos principales y sus responsabilidades son:

| Archivo | Responsabilidad |
|---|---|
| `src/frontend/app.py` | Carga, validación previa, vista previa, progreso, errores y presentación. |
| `src/frontend/client.py` | Envío HTTP de los bytes de imagen y recepción de la respuesta JSON. |
| `src/backend/api.py` | Inicio del servicio y endpoints HTTP. |
| `src/backend/config.py` | Rutas del modelo y etiquetas, límite de carga y valores predeterminados. |
| `src/backend/image_processing.py` | Validación, conversión RGB, redimensionamiento y tensor de entrada. |
| `src/backend/model_service.py` | Carga del modelo y las etiquetas; ejecución de la inferencia y formato de la respuesta. |

El modelo se carga al iniciar el backend. Si no puede cargarse, `/health` informa que no está disponible y las rutas que requieren inferencia no presentan una predicción de reemplazo.

## 11. Descripción de endpoints

| Método y ruta | Entrada | Respuesta o comportamiento |
|---|---|---|
| `GET /health` | Ninguna | HTTP 200 con `status` (`ready` o `unavailable`) y `available` booleano. La disponibilidad se refleja en el cuerpo. |
| `GET /model/info` | Ninguna | HTTP 200 con `model` (arquitectura reconocida o descriptor Keras genérico) y `categories` en orden. HTTP 503 si el modelo no está listo. |
| `POST /predict` | Archivo multipart en el campo `image` | HTTP 200 con `category`, `state` (`Fresh` o `Rotten`) y `confidence` porcentual. |

El límite predeterminado de archivo es 10 MiB (10.485.760 bytes), además del límite predeterminado de 40.000.000 píxeles. Se admiten JPEG, PNG y WEBP. El servicio responde con:

- **400:** archivo vacío, corrupto, formato no admitido o dimensiones superiores al máximo de píxeles.
- **413:** archivo superior al límite configurado en bytes.
- **503:** modelo no disponible en `/model/info` y `/predict`; `/health` conserva HTTP 200 y marca `available: false`.
- **500:** salida numérica del modelo inválida o fallo durante procesamiento/inferencia. Las salidas se validan por forma `(1, número de categorías)`, tipo numérico real, finitud y rango `[0, 1]`. Si la última capa cargada usa softmax, también se comprueba que la suma sea aproximadamente 1.

Cuando el modelo no está listo, `/health` informa disponibilidad falsa; `/model/info` y `/predict` responden con 503.

Durante el inicio, el servicio registra códigos de diagnóstico para etiquetas ausentes o inválidas, modelo ausente, fallo de carga Keras, dimensiones incompatibles y discrepancia en la cantidad de categorías. El estado interno `load_error` guarda el código; los mensajes HTTP no exponen rutas locales ni trazas.

## 12. Instalación y ejecución

Los comandos se ejecutan en PowerShell desde la raíz del proyecto. Se usa el entorno virtual existente en `.venv`; si aún no existe, créalo primero. La instalación se realiza con las versiones declaradas en `requirements.txt`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

Inicia el backend en una terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.backend.api:app --host 127.0.0.1 --port 8000
```

Inicia el frontend en una segunda terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/frontend/app.py
```

Por defecto, Streamlit publica la interfaz en el puerto 8501 y FastAPI escucha en `http://127.0.0.1:8000`. La documentación interactiva de FastAPI está en `http://127.0.0.1:8000/docs`.

Las variables de entorno disponibles son:

| Variable | Valor predeterminado | Uso |
|---|---|---|
| `FRUITVISION_MODEL_PATH` | `models/fruitvision_ai_model.keras` | Sobrescribe la ruta del modelo. |
| `FRUITVISION_CLASS_NAMES_PATH` | `models/class_names.json` | Sobrescribe la ruta de las etiquetas. |
| `FRUITVISION_MAX_UPLOAD_BYTES` | `10485760` | Configura el máximo de bytes por archivo. |
| `FRUITVISION_MAX_IMAGE_PIXELS` | `40000000` | Configura el máximo de píxeles por imagen. |
| `FRUITVISION_API_URL` | `http://127.0.0.1:8000` | Cambia la URL base usada por el frontend. |

## 13. Pruebas unitarias y de integración

Las pruebas se encuentran en `tests/`:

- `test_image_processing.py`: formatos, RGB, tamaño, tipo de dato, rango y archivos inválidos.
- `test_model_service.py`: selección de categoría, validación de salidas numéricas, carga y diagnóstico con dobles.
- `test_api.py`: estado, información, predicción y códigos HTTP para errores con un doble aislado.
- `test_frontend.py`: envío de bytes, representación de respuesta y manejo de errores de UI con respuestas controladas.
- `test_integration_model.py`: usa el modelo original sin sustituir la predicción; recorre la lógica `process_prediction`, las funciones del cliente y la API mediante un adaptador de transporte en memoria basado en `TestClient`. La imagen sintética se crea durante la prueba.

Para ejecutar toda la suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Para ejecutar únicamente la integración con el modelo real:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_integration_model.py -q
```

## 14. Resultados de verificación

La ejecución completa realizada el **20 de noviembre de 2026** finalizó con **42 pruebas aprobadas**, **0 fallidas** y **1 advertencia** en **10.34 segundos**. La advertencia de Starlette señala que el uso de HTTPX con `starlette.testclient` está deprecado y recomienda `httpx2`; no impidió la ejecución ni causó fallos. El conteo de pruebas puede variar cuando cambia la suite.

En la comprobación local del 20 de noviembre de 2026, el comando documentado inició Uvicorn, `/health` devolvió `available: true`, `/model/info` devolvió EfficientNetB0 y 20 categorías, y el comando de Streamlit respondió HTTP 200. La suite incluyó además una predicción integrada con el modelo real. El [inventario de integridad](artifact-integrity.md) conserva hashes registrados de los modelos, las etiquetas, el notebook y los artefactos locales de entrenamiento y evaluación.

## 15. Limitaciones conocidas

El clasificador está diseñado para las 20 categorías definidas en `models/class_names.json`. No tiene una clase adicional para objetos ajenos a ellas ni un mecanismo de detección de imágenes fuera de distribución (OOD). En consecuencia, una fotografía que no corresponda a una fruta o verdura de esas categorías puede recibir igualmente la predicción de una de las 20 clases.

La confianza mostrada es el score del modelo para una imagen individual. No garantiza certeza y no representa una métrica global de rendimiento. Las métricas de evaluación de este informe describen el conjunto de prueba registrado, no el resultado esperado para cualquier fotografía.

## 16. Conclusiones

FruitVision AI integra una interfaz de carga, una API HTTP y el modelo EfficientNetB0 existente en un flujo de inferencia local. El preprocesamiento documentado conserva el tamaño, los canales, el tipo de dato y el rango verificados, dejando las capas internas de normalización a cargo del modelo.

La evaluación existente alcanzó los valores detallados en `reports/metrics.json`, y la suite ejecutada el 20 de noviembre de 2026 aprobó 42 pruebas. El uso del sistema debe considerar el límite de sus 20 clases y la interpretación acotada de la puntuación individual.
