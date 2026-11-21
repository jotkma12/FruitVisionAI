# Tasks

> **Registro histórico:** esta lista conserva las tareas y afirmaciones de verificación del cambio archivado el 19 de noviembre de 2026. Los checks reflejan ese registro histórico y no sustituyen una ejecución actual; consulta `openspec/specs/`, `README.md` y `docs/technical.md` para el estado vigente.

## 1. Verificación previa y estructura

- [x] 1.1 Consultar en modo de solo lectura `FruitVisionAI/fruitvision_ai_entrenamiento.ipynb` (sin ejecutar celdas) y verificar en su código el tamaño de entrada, la conversión a RGB, el rango de valores y la normalización realmente usados durante el entrenamiento; registrar la evidencia en la documentación técnica y hacer que el diseño de inferencia coincida antes de codificar el procesamiento. Si algún dato no puede confirmarse, detener la integración de inferencia hasta resolverlo. Conservar intactos el notebook y todos los artefactos existentes; no ejecutarlo, editarlo, reconstruirlo ni reemplazarlo.
- [x] 1.2 Verificar en el entorno local la versión de Python y la compatibilidad de TensorFlow/Keras cargando el modelo original sin guardarlo ni modificarlo; registrar la combinación compatible y comprobar que el archivo permanece intacto.
- [x] 1.3 Añadir los archivos de aplicación, pruebas y documentación dentro de las carpetas existentes `src/backend/`, `src/frontend/`, `tests/` y `docs/`, junto con configuración de rutas y comunicación; comprobar que las rutas predeterminadas resuelven los archivos originales de `models/` sin copiarlos y que no se crearon carpetas paralelas.
- [x] 1.4 Declarar y fijar las dependencias de aplicación y pruebas según las verificaciones anteriores; comprobar la instalación en un entorno limpio compatible.

## 2. Servicio de inferencia

- [x] 2.1 Implementar carga única del modelo y carga de etiquetas desde JSON, verificando compatibilidad de dimensiones y readiness; comprobar con pruebas que errores de carga no se reportan como servicio disponible.
- [x] 2.2 Implementar decodificación, validación de tipo/tamaño, conversión RGB, redimensionamiento y preprocesamiento confirmado; comprobar formatos permitidos, archivos vacíos/corruptos, límite configurado y forma del tensor.
- [x] 2.3 Implementar `GET /health`, `GET /model/info` y `POST /predict` con respuesta de categoría, estado y confianza derivada del modelo; comprobar contratos exitosos y fallidos con HTTPX y dobles solo en pruebas aisladas.
- [x] 2.4 Documentar contrato de API, formatos, tamaño máximo, errores, significado de confianza y comandos para iniciar el servicio; verificar que las rutas y ejemplos coinciden con el servicio implementado.

## 3. Interfaz de usuario

- [x] 3.1 Implementar carga y vista previa de fotografía, estado de procesamiento y presentación de resultados usando el servicio HTTP; comprobar que una carga nueva envía sus bytes y que la UI representa la respuesta recibida.
- [x] 3.2 Implementar mensajes para formato inválido, error de conexión y fallo de inferencia sin conservar ni sustituir por una predicción anterior; comprobar cada estado con pruebas del cliente/interfaz.
- [x] 3.3 Documentar cómo iniciar la interfaz y configurar la URL del servicio; verificar el flujo de arranque y la comunicación con la API siguiendo las instrucciones.

## 4. Documentación y verificación integrada

- [x] 4.1 Escribir README y documento técnico con arquitectura, lenguajes, frameworks, librerías, justificación, requisitos e instalación; documentar el notebook existente en `FruitVisionAI/fruitvision_ai_entrenamiento.ipynb` como fuente consultada en modo de solo lectura, sin ejecutarlo; verificar que los comandos documentados funcionan en el entorno compatible.
- [x] 4.2 Añadir prueba de integración que envíe una imagen de prueba al servicio y cargue el modelo original real, sin mocks ni expectativas de una etiqueta precalculada; verificar que produce una respuesta válida con categoría del JSON y confianza numérica.
- [x] 4.3 Preparar guía de demostración y sustentación con pasos para seleccionar una fotografía durante la ejecución, seguir la solicitud hasta el modelo y mostrar la respuesta devuelta; verificar que no usa predicciones precargadas o simuladas.
- [x] 4.4 Revisar que las pruebas unitarias, de API, de interfaz e integración sean reproducibles y que la documentación distinga dobles de prueba del uso real del modelo; ejecutar los comandos documentados y registrar resultados observados.
- [x] 4.5 Comprobar que modelos, etiquetas y artefactos locales de entrenamiento/evaluación permanecen sin cambios mediante hashes antes/después o evidencia equivalente; registrar `FruitVisionAI/fruitvision_ai_entrenamiento.ipynb` como ubicación del notebook existente y comprobar que permanece sin modificaciones mediante hash antes/después, sin ejecución, edición, reconstrucción ni reemplazo.
