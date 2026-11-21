# Design

> **Registro histórico:** este diseño conserva las decisiones y supuestos de planificación del cambio archivado el 19 de noviembre de 2026. No es una descripción normativa del código actual; consulta `openspec/specs/`, `README.md` y `docs/technical.md` para el estado vigente.

## Context

Ver `proposal.md` y las capacidades `live-inference`, `fruitvision-user-experience` y `technical-delivery`. El workspace ya contiene `src/backend/`, `src/frontend/`, `tests/` y `docs/`, que se reutilizarán; no se crearán carpetas paralelas. El artefacto de modelo y el JSON requeridos existen en `models/`; existen copias dentro de `FruitVisionAI/` que forman parte de los entregables terminados y se conservarán. El notebook de entrenamiento existente está localizado en `FruitVisionAI/fruitvision_ai_entrenamiento.ipynb`. Se consultará su contenido en modo de solo lectura para verificar el preprocesamiento real; no se ejecutará, editará, reconstruirá ni reemplazará.

## Goals / Non-Goals

**Goals:**

- Separar interfaz, transporte HTTP e inferencia para que cada componente se pueda ejecutar y comprobar.
- Hacer demostrable el trayecto fotografía cargada → API → modelo guardado → respuesta → interfaz.
- Elegir dependencias por su adecuación al modelo Keras existente y a los entregables requeridos.

**Non-Goals:**

- Entrenar, convertir, evaluar nuevamente o editar el modelo, las categorías, el notebook o los artefactos existentes.
- Agregar despliegue en nube, persistencia, cuentas de usuario o un sistema de monitoreo no requerido por la rúbrica.
- Afirmar que una predicción individual es certeza diagnóstica.

## Decisions

### Python con TensorFlow/Keras para la inferencia

La aplicación usará Python porque el archivo `.keras` fue exportado para TensorFlow/Keras y permite cargarlo directamente sin conversión. Se validará la compatibilidad de versión antes de fijar dependencias; las versiones se documentarán y fijarán en el manifiesto de ejecución. No se instalará un segundo runtime ni se reexportará el modelo.

### FastAPI como servicio y Streamlit como interfaz

Se propone FastAPI para exponer un contrato HTTP sencillo (`GET /health`, `GET /model/info`, `POST /predict`), validar cargas y permitir pruebas de solicitudes con HTTPX. Se propone Streamlit para ofrecer carga de imagen, vista previa y resultado en Python, evitando introducir un frontend JavaScript separado. La combinación requiere dos procesos, pero hace visible la integración que debe demostrarse. Una interfaz Streamlit que ejecutara el modelo directamente simplificaría el arranque, aunque dejaría sin demostrar la comunicación entre componentes; una SPA web exigiría otra cadena de herramientas sin necesidad indicada.

La API cargará modelo y etiquetas al inicio, mantendrá una instancia reutilizable y devolverá un contrato estable con categoría, estado y confianza. La interfaz enviará los bytes de cada carga nueva al endpoint y no mantendrá predicciones de respaldo. El backend será la única fuente de resultados.

### Fuente de modelo y etiquetas

La configuración predeterminada apuntará a `models/fruitvision_ai_model.keras` y `models/class_names.json`, las rutas que el usuario señaló como entregables de integración. Se podrán configurar rutas para facilitar ejecución en otros entornos. No se copiarán ni modificarán los artefactos: tampoco se elegirán ni eliminarán las copias bajo `FruitVisionAI/` sin una razón técnica documentada.

El JSON será la única fuente de nombres y orden de categorías en ejecución. El arranque comprobará que su cantidad coincide con la dimensión de salida del modelo. Las categorías Fresh/Rotten y la confianza se derivarán de la salida real.

### Preprocesamiento como condición de compatibilidad

Durante la planificación archivada se propuso consultar el notebook en modo de solo lectura y sin ejecutar celdas. La verificación documentada entonces describía una entrada de 224 × 224, tres canales RGB y conversión a `float32`; el modelo incorpora sus propias capas de escalado y normalización. La ruta citada en ese plan (`FruitVisionAI/fruitvision_ai_entrenamiento.ipynb`) era un supuesto histórico: el notebook presente en el repositorio está en `notebooks/fruitvision_ai_entrenamiento.ipynb`. La implementación vigente y sus límites se documentan en `docs/technical.md`.

### Pillow para decodificación de imágenes

Se propone Pillow para reconocer y decodificar los formatos admitidos, detectar archivos dañados, convertir a RGB y redimensionar. Es una dependencia Python acotada y suficiente para este flujo; OpenCV se considera innecesario porque no se requieren operaciones de visión adicionales. El tamaño máximo de carga se configurará explícitamente y se documentará junto a los formatos aceptados.

### Pruebas por frontera y recorrido real

Se propone pytest para pruebas reproducibles y HTTPX para comprobar la API y el cliente HTTP. Las pruebas unitarias podrán aislar el modelo para verificar errores y validaciones. Una prueba de integración separada cargará el modelo real y enviará una imagen de prueba a través del endpoint; la demostración manual repetirá el flujo desde Streamlit con una fotografía escogida durante la ejecución. La documentación distinguirá claramente la prueba con modelo real de las pruebas con dobles.

### Estructura propuesta

- `src/backend/`: API FastAPI, carga del modelo, etiquetas, procesamiento e inferencia.
- `src/frontend/`: interfaz Streamlit y cliente HTTP del servicio.
- `tests/`: pruebas de unidad, API, interfaz e integración.
- `docs/`: arquitectura, decisiones y guía de demostración/sustentación.
- `README.md` y manifiesto de dependencias en la raíz.
- `models/`: entradas de solo lectura para el modelo y el JSON de categorías.

Estas carpetas ya existen en el workspace; la implementación añadirá los archivos dentro de ellas y evitará crear duplicados como `src/fruitvision/`, `frontend/`, `backend/` o nuevas carpetas de pruebas/documentación en la raíz.

## Risks / Trade-offs

- [El notebook no debe alterarse durante la verificación] → Consultar `FruitVisionAI/fruitvision_ai_entrenamiento.ipynb` en modo de solo lectura y sin ejecutar celdas; verificar allí el preprocesamiento real. No reconstruirlo, editarlo ni reemplazarlo, y conservar todos los artefactos existentes.
- [Versiones TensorFlow/Keras incompatibles con el artefacto] → Verificar la carga del archivo original en el entorno elegido y fijar la versión compatible sin reexportar el modelo.
- [Carga inicial y memoria del modelo] → Cargar una vez al iniciar el backend e informar readiness; no cargarlo por solicitud.
- [Prueba integrada dependiente del modelo y de una imagen disponible] → Mantenerla separada de la suite rápida y documentar su requisito de archivos y ejecución; no convertirla en una prueba con predicción fija.
- [Dos procesos añaden pasos de inicio] → Proveer comandos claros para ejecutar API e interfaz y una guía de verificación de conectividad.

## Migration Plan

No hay migración de datos ni cambio de formato del modelo. La implementación añadirá código, dependencias, documentación y configuración. La ejecución se podrá revertir retirando esos componentes nuevos; los entregables de entrenamiento permanecen fuera de esa operación.
