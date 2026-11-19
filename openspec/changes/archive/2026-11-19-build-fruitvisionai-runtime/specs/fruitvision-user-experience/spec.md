# Spec Delta

## Purpose

Permitir que una persona cargue una fotografía y consulte en una interfaz clara el resultado que el modelo calculó para esa imagen durante la sesión.

## ADDED Requirements

### Requirement: El usuario puede cargar una fotografía
La interfaz MUST permitir seleccionar una imagen admitida desde el dispositivo y mostrar una vista previa antes de solicitar la predicción.

#### Scenario: Imagen seleccionada
- **WHEN** el usuario selecciona una imagen admitida
- **THEN** la interfaz muestra su vista previa y habilita la acción para analizarla

#### Scenario: Archivo no admitido
- **WHEN** el usuario selecciona un archivo no admitido o vacío
- **THEN** la interfaz explica el problema y no presenta una predicción

### Requirement: La interfaz consulta el servicio de inferencia
La interfaz MUST enviar los bytes de la imagen seleccionada al servicio de inferencia y MUST presentar únicamente el resultado recibido para esa solicitud.

#### Scenario: Solicitud procesada
- **WHEN** el usuario solicita analizar una imagen válida y el servicio responde correctamente
- **THEN** la interfaz presenta la respuesta del servicio asociada a esa imagen

#### Scenario: Servicio inaccesible
- **WHEN** la interfaz no puede conectarse al servicio o recibe un error
- **THEN** muestra un mensaje útil y no sustituye el resultado por una predicción local, simulada o anterior

### Requirement: La interfaz presenta el resultado completo
La interfaz MUST mostrar la categoría devuelta, el estado fresco o podrido y la confianza en porcentaje, con la aclaración sobre el alcance de la confianza.

#### Scenario: Mostrar predicción
- **WHEN** el servicio devuelve una predicción válida
- **THEN** la interfaz muestra la fruta o verdura, su estado, el porcentaje de confianza y la imagen analizada

### Requirement: La interfaz informa el progreso de análisis
La interfaz MUST comunicar que la imagen está siendo procesada mientras espera la respuesta y MUST permitir distinguir el progreso de un resultado final.

#### Scenario: Inferencia en curso
- **WHEN** la solicitud está pendiente
- **THEN** la interfaz muestra un estado de procesamiento y no presenta todavía un resultado como final
