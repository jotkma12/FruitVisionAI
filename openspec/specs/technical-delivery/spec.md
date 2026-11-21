# technical-delivery Specification

## Purpose

Hacer que la aplicación pueda instalarse, ejecutarse, verificarse y sustentarse técnicamente con instrucciones reproducibles y evidencia que corresponda al comportamiento real.

## Requirements

### Requirement: La arquitectura y las tecnologías quedan justificadas
La documentación MUST describir la arquitectura, los componentes, el lenguaje, frameworks y librerías seleccionados, y justificar su elección frente a alternativas relevantes para este proyecto.

#### Scenario: Revisión técnica
- **WHEN** un revisor consulta el documento técnico
- **THEN** puede identificar responsabilidades, flujo entre componentes, tecnologías y razones de selección

### Requirement: La aplicación incluye instrucciones reproducibles
El proyecto MUST documentar requisitos previos, instalación de dependencias, configuración necesaria, inicio de los componentes y consulta de los endpoints o interfaz disponibles.

#### Scenario: Preparar una ejecución local
- **WHEN** una persona sigue el README desde un entorno limpio compatible
- **THEN** puede configurar e iniciar la aplicación siguiendo los pasos documentados

### Requirement: Las dependencias y la configuración corresponden al código
El proyecto MUST declarar las dependencias necesarias y documentar la configuración de rutas del modelo, etiquetas, URL del backend, límite de bytes y límite de píxeles (`FRUITVISION_MAX_IMAGE_PIXELS`, predeterminado 40.000.000) sin incrustar credenciales ni depender de resultados de predicción precargados.

#### Scenario: Instalación declarada
- **WHEN** se instalan las dependencias desde el manifiesto del proyecto
- **THEN** quedan disponibles las librerías requeridas por la aplicación y sus pruebas

### Requirement: Las pruebas cubren componentes y flujo integrado
El proyecto MUST incluir pruebas reproducibles para validación y respuesta de inferencia, interfaz/cliente y el recorrido integrado de una fotografía hasta el resultado. Las pruebas aisladas pueden sustituir dependencias externas, pero la prueba integrada MUST ejercitar el modelo real. La prueba integrada actual pasa por `process_prediction` y las funciones del cliente hacia FastAPI mediante un adaptador en memoria respaldado por `TestClient`; no simula la predicción del modelo.

#### Scenario: Prueba aislada de error
- **WHEN** se ejecutan las pruebas de componente con entradas inválidas o fallos simulados
- **THEN** se comprueban las respuestas de error esperadas sin modificar el modelo ni los artefactos de entrenamiento

#### Scenario: Prueba integrada con modelo real
- **WHEN** se ejecuta la prueba de integración con una imagen de prueba disponible y el modelo exportado
- **THEN** la imagen atraviesa la lógica de procesamiento del frontend y el cliente, el servicio y el modelo real, y se recibe una respuesta de inferencia; la representación Streamlit se cubre aparte con pruebas de UI

### Requirement: Los entregables de entrenamiento se preservan
El desarrollo MUST dejar sin cambios el notebook terminado, los modelos, las categorías y los artefactos de entrenamiento y evaluación; MUST NOT volver a entrenar ni regenerar estos entregables. El inventario vigente se organiza en `notebooks/`, `data/processed/`, `models/training/` y `reports/`; el modelo oficial de produccion permanece en `models/fruitvision_ai_model.keras` junto con `models/class_names.json`.

#### Scenario: Implementación de la aplicación
- **WHEN** se añaden o actualizan componentes de aplicación y documentación
- **THEN** los archivos existentes del entrenamiento permanecen intactos y el notebook no se reconstruye

### Requirement: La demostración prueba inferencia en vivo
La guía de demostración MUST describir cómo cargar una fotografía durante la ejecución, observar su envío al servicio y comprobar que la respuesta proviene del modelo entrenado y corresponde a esa solicitud.

#### Scenario: Sustentación con imagen cargada
- **WHEN** se realiza la demostración en vivo con una fotografía seleccionada en ese momento
- **THEN** se muestra el flujo interfaz-servicio-modelo y el resultado generado para dicha imagen, sin depender de predicciones fijas o simuladas

### Requirement: La documentación distingue métricas de entrenamiento y predicción
La documentación MUST atribuir las métricas existentes a la evaluación ya realizada y MUST distinguirlas de la confianza devuelta para una imagen individual.

#### Scenario: Consulta de resultados reportados
- **WHEN** la documentación presenta métricas del modelo
- **THEN** las identifica como resultados del conjunto de evaluación existente y no como garantía de una predicción individual
