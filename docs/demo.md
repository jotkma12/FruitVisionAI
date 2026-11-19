# Demostración y sustentación

## Preparación

1. Activa el entorno virtual e instala las versiones de `requirements.txt` según el README.
2. En una terminal, desde la raíz, inicia la API:

   ```powershell
   python -m uvicorn src.backend.api:app --reload
   ```

3. Comprueba `http://127.0.0.1:8000/health`: debe responder `available: true`. Abre `/model/info` para mostrar EfficientNetB0 y las categorías que cargó desde `models/class_names.json`.
4. En otra terminal, inicia la interfaz:

   ```powershell
   python -m streamlit run src/frontend/app.py
   ```

5. Abre la URL local de Streamlit indicada en la terminal.

## Recorrido de una imagen en vivo

1. Selecciona una fotografía JPEG, PNG o WEBP en la interfaz y muestra la vista previa.
2. Pulsa **Analizar imagen**. La interfaz muestra el estado de procesamiento mientras espera la respuesta.
3. Sigue el recorrido en el código: `src/frontend/app.py` obtiene los bytes de la carga; `src/frontend/client.py` envía esos bytes en `POST /predict`; `src/backend/api.py` recibe el archivo; `src/backend/image_processing.py` valida, convierte a RGB y prepara el tensor; `src/backend/model_service.py` llama a la instancia del modelo real.
4. Muestra en la terminal de Uvicorn la línea `POST /predict ... 200 OK` y en la interfaz la categoría, el estado y la puntuación devueltos por esa solicitud.
5. Selecciona una segunda fotografía durante la demostración. La interfaz elimina el resultado anterior al cambiar los bytes y presenta el resultado que devuelve la API para la nueva imagen.

El servicio no carga resultados precalculados: ejecuta el modelo para cada carga válida. `/docs` permite inspeccionar y enviar una petición de API, y `/health` permite mostrar el estado de disponibilidad. La confianza es la puntuación del modelo para esa imagen, no una certeza absoluta ni una métrica global.

## Evidencia técnica

`tests/test_integration_model.py` ejerce el servicio con el modelo original sin mocks. Las pruebas de interfaz usan dobles solo para aislar respuestas de API y errores; no se usan esos dobles en la demostración en vivo. El notebook de entrenamiento y todos los artefactos de entrenamiento se conservan sin cambios.

## Ubicacion de los entregables

El notebook esta en `notebooks/fruitvision_ai_entrenamiento.ipynb`; los datos procesados en `data/processed/`; los modelos de entrenamiento en `models/training/`; y los informes y metricas en `reports/`. La demostracion usa por defecto el modelo oficial `models/fruitvision_ai_model.keras` y las etiquetas `models/class_names.json`.
