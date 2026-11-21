# Proposal

> **Registro histórico:** esta propuesta describe la planificación del cambio archivado el 19 de noviembre de 2026. Sus supuestos iniciales no describen necesariamente el árbol actual. Consulta `openspec/specs/`, `README.md` y `docs/technical.md` para el estado vigente.

## Why

El entrenamiento y la evaluación ya están terminados, y el workspace contiene el modelo exportado y sus etiquetas, pero todavía no hay una aplicación que produzca y muestre una predicción a partir de una fotografía cargada en vivo. Este cambio define y entrega la capa de aplicación, sus pruebas y la documentación necesaria para demostrar una inferencia real sin rehacer ni alterar los entregables de entrenamiento.

## What Changes

- Crear una aplicación que acepte una fotografía del usuario durante la ejecución y presente la categoría, el estado fresco/podrido y la confianza que devuelve el modelo real.
- Integrar el modelo EfficientNetB0 exportado y el JSON de categorías, con validación de archivos, errores comprensibles y consultas de estado e información.
- Especificar y comprobar el flujo integrado desde la interfaz hasta la inferencia, incluyendo una demostración en vivo sin predicciones simuladas o precargadas.
- Documentar la arquitectura y justificar tecnologías, lenguajes, frameworks y librerías seleccionadas; incluir instrucciones, dependencias y configuración de ejecución.
- Mantener pruebas de componentes e integración reproducibles y documentar la sustentación técnica.
- Conservar el notebook de entrenamiento ya terminado en Google Colab como entregable académico existente y documentar esa ubicación; no reconstruirlo, desarrollarlo ni modificarlo. Excluir también el reentrenamiento, la evaluación experimental y toda edición o regeneración del modelo, las etiquetas y los artefactos existentes.

## Capabilities

### New Capabilities

- `live-inference`: carga y consulta del modelo entrenado, categorías, recepción y validación de imágenes, inferencia y respuesta real de clasificación.
- `fruitvision-user-experience`: carga de imagen, vista previa, comunicación con inferencia, presentación de resultados y manejo visible de errores.
- `technical-delivery`: documentación técnica y de ejecución, pruebas reproducibles y demostración del flujo real integrado.

### Modified Capabilities

Ninguna. El proyecto no tiene especificaciones de capacidades existentes.

## Impact

- El workspace ya contiene `src/backend/`, `src/frontend/`, `tests/` y `docs/`, pero no se encontraron archivos de aplicación en esas carpetas ni README o manifiesto de dependencias en la raíz. La implementación aprobada deberá completar esas carpetas existentes, no crear duplicados.
- Los recursos indicados para inferencia son `models/fruitvision_ai_model.keras` y `models/class_names.json`, tratados como entradas de solo lectura. También existe una copia en `FruitVisionAI/`; ambas copias y todos los CSV, JSON, imágenes y modelos de entrenamiento se conservarán sin alteración.
- El notebook terminado existe en Google Colab según lo indicado por el usuario. Se documentará su ubicación como entregable existente; no se reconstruirá, descargará para reemplazar archivos locales ni editará.
- La propuesta contempla una interfaz y un servicio conectados para hacer observable la integración completa; el diseño comparará opciones y justificará cada tecnología antes de fijar dependencias.
