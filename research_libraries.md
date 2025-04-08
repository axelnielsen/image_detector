# Investigación de Bibliotecas para Procesamiento de Imágenes

## Bibliotecas principales

### OpenCV
- **Descripción**: Biblioteca de visión por computadora de código abierto con más de 2500 algoritmos optimizados.
- **Ventajas**: 
  - Funciones para procesamiento básico de imágenes (filtros, transformaciones)
  - Detección de características (bordes, esquinas)
  - Segmentación de imágenes
  - Detección de objetos mediante clasificadores en cascada (Haar, LBP)
  - Operaciones morfológicas útiles para análisis de fondo
- **Aplicación en nuestro proyecto**:
  - Detección de fondo blanco mediante análisis de histograma y umbralización
  - Pre-procesamiento de imágenes para otras tareas

### TensorFlow/Keras
- **Descripción**: Framework de aprendizaje profundo desarrollado por Google.
- **Ventajas**:
  - Modelos pre-entrenados para detección de objetos (SSD, Faster R-CNN, EfficientDet)
  - Transfer learning para adaptar modelos a nuestras necesidades específicas
  - Flexibilidad para crear arquitecturas personalizadas
- **Aplicación en nuestro proyecto**:
  - Detección de personas mediante modelos pre-entrenados
  - Clasificación de tipos de documentos (cédula vs otros)

### scikit-learn
- **Descripción**: Biblioteca de aprendizaje automático para Python.
- **Ventajas**:
  - Algoritmos de clasificación (SVM, Random Forest, etc.)
  - Herramientas para evaluación de modelos
  - Preprocesamiento de datos
- **Aplicación en nuestro proyecto**:
  - Clasificación de características extraídas de las imágenes
  - Evaluación del rendimiento de los modelos

### scikit-image
- **Descripción**: Colección de algoritmos para procesamiento de imágenes.
- **Ventajas**:
  - Segmentación de imágenes
  - Transformaciones geométricas
  - Análisis de textura
- **Aplicación en nuestro proyecto**:
  - Complemento a OpenCV para tareas específicas de procesamiento

## Enfoques para cada característica a detectar

### 1. Detección de fondo blanco
- **Enfoque recomendado**: Análisis de histograma y umbralización con OpenCV
- **Técnicas**:
  - Conversión a escala de grises
  - Análisis de histograma para determinar distribución de colores
  - Umbralización para separar fondo de primer plano
  - Cálculo de porcentaje de píxeles blancos
- **Bibliotecas**: OpenCV, scikit-image

### 2. Detección de personas
- **Enfoque recomendado**: Modelo de deep learning pre-entrenado
- **Técnicas**:
  - Utilizar modelos como YOLO, SSD o Faster R-CNN pre-entrenados en COCO dataset
  - Fine-tuning para mejorar precisión en nuestro contexto
  - Extracción de características faciales para verificación adicional
- **Bibliotecas**: TensorFlow/Keras, OpenCV (para HOG + SVM como alternativa más ligera)

### 3. Detección de cédula de identidad
- **Enfoque recomendado**: Combinación de detección de bordes y clasificación
- **Técnicas**:
  - Detección de contornos rectangulares
  - Extracción de características (relación de aspecto, textura)
  - Clasificador entrenado para distinguir cédulas de otros documentos
  - Posible uso de template matching para identificar elementos específicos de cédulas chilenas
- **Bibliotecas**: OpenCV, TensorFlow/Keras

### 4. Detección de RUT chileno
- **Enfoque recomendado**: OCR + validación de patrón
- **Técnicas**:
  - Preprocesamiento para mejorar calidad de texto
  - OCR para extraer texto de la imagen
  - Expresiones regulares para identificar patrones de RUT (XX.XXX.XXX-X)
  - Validación del dígito verificador según algoritmo chileno
- **Bibliotecas**: Tesseract OCR (a través de pytesseract), OpenCV para preprocesamiento

## Conclusiones y recomendaciones

Para nuestro sistema, se recomienda un enfoque híbrido que combine:

1. **Técnicas clásicas de visión por computadora** (OpenCV) para tareas como detección de fondo blanco y preprocesamiento.
2. **Modelos de deep learning** (TensorFlow/Keras) para detección de personas y clasificación de documentos.
3. **OCR** para la extracción y validación de RUT.

Este enfoque nos permitirá balancear precisión y eficiencia, adaptando la complejidad de cada método a la tarea específica.
