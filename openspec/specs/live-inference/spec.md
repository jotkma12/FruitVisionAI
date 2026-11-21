# live-inference Specification

## Purpose

Define el servicio que carga el modelo Keras configurado para clasificar en tiempo de ejecución imágenes del usuario y devuelve resultados trazables a las categorías entrenadas.

## Requirements

### Requirement: El servicio reutiliza el modelo entrenado
El servicio MUST cargar el modelo existente una vez al iniciar y reutilizarlo para las solicitudes de inferencia. MUST tratar el archivo del modelo como entrada de solo lectura y MUST reportar el servicio como no disponible cuando no pueda cargarlo.

#### Scenario: Modelo cargado al iniciar
- **WHEN** el servicio inicia con un modelo accesible y válido
- **THEN** el servicio queda disponible para inferencia y reutiliza esa instancia en solicitudes posteriores

#### Scenario: Fallo al cargar el modelo
- **WHEN** el servicio no puede leer o cargar el modelo configurado
- **THEN** informa que la inferencia no está disponible y no devuelve una predicción de reemplazo

### Requirement: Las categorías provienen del archivo autorizado
El servicio MUST cargar los nombres y el orden de categorías desde `models/class_names.json` y MUST verificar su compatibilidad con las salidas del modelo. MUST conservar las 20 etiquetas existentes en su orden, sin mantener una lista alternativa codificada.

#### Scenario: Categorías compatibles
- **WHEN** el archivo de categorías contiene las etiquetas esperadas y coincide con la salida del modelo
- **THEN** la respuesta usa el nombre asociado al índice predicho por ese archivo

#### Scenario: Categorías ausentes o incompatibles
- **WHEN** el archivo no puede cargarse o su cantidad de etiquetas no coincide con la salida del modelo
- **THEN** el servicio informa un error de configuración y no presenta una etiqueta inferida de una lista diferente

### Requirement: El servicio valida las imágenes recibidas
El servicio MUST aceptar JPEG, PNG y WEBP y MUST rechazar cargas vacías, inválidas, mayores que el límite de bytes configurado o superiores al límite de píxeles configurado. El valor predeterminado del límite de píxeles es 40.000.000 y se configura mediante `FRUITVISION_MAX_IMAGE_PIXELS`.

#### Scenario: Imagen admitida
- **WHEN** se recibe una imagen válida en un formato permitido y dentro del límite configurado
- **THEN** el servicio la acepta para procesamiento

#### Scenario: Imagen no válida o fuera de límite
- **WHEN** se recibe un archivo vacío, corrupto, de tipo no admitido o mayor que el límite documentado
- **THEN** el servicio rechaza la solicitud con un error identificable y no ejecuta inferencia

#### Scenario: Imagen con dimensiones excesivas
- **WHEN** el ancho por el alto supera el límite de píxeles configurado
- **THEN** la API responde HTTP 400 antes de convertir o redimensionar la imagen

### Requirement: El procesamiento conserva la compatibilidad del entrenamiento
El servicio MUST convertir la imagen recibida a RGB, preparar la entrada a la dimensión esperada de 224 × 224 y aplicar el preprocesamiento verificado como compatible con el entrenamiento existente.

#### Scenario: Imagen válida preparada para inferencia
- **WHEN** se acepta una imagen de usuario
- **THEN** el modelo recibe un tensor RGB con la forma y el preprocesamiento compatibles con el entrenamiento

### Requirement: La predicción se calcula en el momento
Para cada solicitud válida, el servicio MUST ejecutar el modelo entrenado sobre la imagen recibida y devolver la categoría de mayor puntuación, su estado Fresh o Rotten derivado del prefijo, y la puntuación correspondiente como porcentaje de confianza.

#### Scenario: Predicción de producto fresco
- **WHEN** el modelo produce como índice máximo una categoría cuyo nombre empieza por `Fresh`
- **THEN** la respuesta contiene esa categoría, el estado fresco y la puntuación del modelo expresada como porcentaje

#### Scenario: Predicción de producto podrido
- **WHEN** el modelo produce como índice máximo una categoría cuyo nombre empieza por `Rotten`
- **THEN** la respuesta contiene esa categoría, el estado podrido y la puntuación del modelo expresada como porcentaje

### Requirement: Las salidas del modelo se validan antes de responder
El servicio MUST exigir una salida numérica real de forma `(1, número de categorías)`, con valores finitos en `[0, 1]`. Si la última capa del modelo cargado usa softmax, MUST comprobar que la suma sea aproximadamente 1. MUST rechazar una salida inválida sin generar categoría ni porcentaje.

#### Scenario: El modelo devuelve valores inválidos
- **WHEN** la salida tiene forma incorrecta, contiene valores no numéricos, no finitos, fuera de rango o no suma aproximadamente 1 para softmax
- **THEN** la API responde HTTP 500 con un mensaje controlado y no devuelve una predicción

### Requirement: La confianza se comunica con su alcance correcto
La aplicación MUST presentar la puntuación de salida asociada a la predicción como confianza del modelo y MUST aclarar que no representa certeza absoluta ni el rendimiento global del modelo.

#### Scenario: Resultado mostrado
- **WHEN** se muestra la confianza de una predicción
- **THEN** se muestra en porcentaje junto a una aclaración de que corresponde a la salida del modelo para esa imagen

### Requirement: El servicio expone estado e información del modelo
El servicio MUST ofrecer `GET /health` para consultar disponibilidad, `GET /model/info` para consultar la identificación del modelo cargado y las categorías, y `POST /predict` para recibir una imagen y devolver la inferencia.

#### Scenario: Servicio saludable
- **WHEN** se consulta el estado con modelo y etiquetas listos
- **THEN** el servicio informa que está disponible

#### Scenario: Información del modelo disponible
- **WHEN** se consulta la información con el servicio inicializado
- **THEN** se devuelve la arquitectura si está reconocida (EfficientNetB0 para el artefacto actual) o un descriptor Keras genérico, junto con las categorías cargadas en su orden

### Requirement: Los errores de carga quedan diagnosticados sin exponer rutas
El inicio MUST registrar códigos que distingan etiquetas ausentes o inválidas, modelo ausente, fallo al cargar el modelo, dimensiones incompatibles y discrepancia en la cantidad de clases. El estado de salud MUST reflejar que el modelo no está disponible y las rutas de inferencia MUST responder HTTP 503. Los mensajes HTTP MUST NOT exponer rutas locales ni trazas.

#### Scenario: Error durante el inicio
- **WHEN** falla la carga o la validación del modelo o las etiquetas
- **THEN** `load_error` conserva un código diagnóstico, los logs registran la categoría del fallo y la API no expone detalles locales

#### Scenario: Predicción por API
- **WHEN** se envía una imagen válida a `POST /predict`
- **THEN** el servicio devuelve la categoría, el estado y la confianza calculados para esa imagen

### Requirement: Los errores de inferencia son explícitos
El servicio MUST convertir fallos de procesamiento o de inferencia en respuestas de error comprensibles y MUST NOT devolver una predicción simulada o precargada.

#### Scenario: Fallo durante la inferencia
- **WHEN** el modelo produce una excepción al procesar una imagen válida
- **THEN** el servicio informa el fallo sin inventar categoría ni confianza
