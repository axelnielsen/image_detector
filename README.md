# Sistema de Detección y Validación de Imágenes

## Descripción General

Este sistema permite entrenar modelos para detectar características específicas en imágenes, como:
- Fondo blanco
- Presencia de personas
- Imágenes tipo cédula de identidad
- RUT chileno visible

El sistema está compuesto por módulos independientes para cada tipo de detección, un pipeline de entrenamiento integrado y un sistema de validación que genera informes detallados.

## Estructura del Proyecto

```
detector_imagenes/
├── dataset/                      # Estructura de datos para entrenamiento
│   ├── train/                    # Datos de entrenamiento
│   └── test/                     # Datos de prueba
├── models/                       # Modelos entrenados
├── test_images/                  # Imágenes de prueba
├── test_results/                 # Resultados de las pruebas
├── validation_results/           # Informes de validación
├── fondo_blanco_detector.py      # Detector de fondo blanco
├── persona_detector.py           # Detector de personas
├── cedula_detector.py            # Detector de cédulas de identidad
├── rut_detector.py               # Detector de RUT chileno
├── model_trainer.py              # Pipeline de entrenamiento
├── image_validator.py            # Sistema de validación
├── dataset_manager.py            # Gestor de conjunto de datos
├── create_test_images.py         # Generador de imágenes de prueba
└── test_system.py                # Script de prueba y evaluación
```

## Componentes Principales

### 1. Detectores Individuales

#### Detector de Fondo Blanco
- Utiliza análisis de histograma y umbralización para determinar si una imagen tiene fondo blanco.
- Calcula el porcentaje de píxeles blancos y determina si supera un umbral configurable.
- Rendimiento: F1 Score de 0.80 en pruebas.

#### Detector de Personas
- Utiliza un modelo pre-entrenado MobileNetV2 para detectar personas en imágenes.
- Identifica clases relacionadas con personas en la clasificación de ImageNet.
- Rendimiento: F1 Score de 0.00 en pruebas con imágenes sintéticas (requiere mejoras).

#### Detector de Cédulas de Identidad
- Utiliza detección de contornos y análisis de proporciones para identificar documentos tipo cédula.
- Verifica características como relación de aspecto y área relativa.
- Rendimiento: F1 Score de 1.00 en pruebas.

#### Detector de RUT Chileno
- Utiliza OCR (Tesseract) para extraer texto de la imagen.
- Aplica expresiones regulares para identificar patrones de RUT chileno.
- Valida el dígito verificador según el algoritmo chileno.
- Rendimiento: F1 Score de 0.33 en pruebas.

### 2. Pipeline de Entrenamiento

El módulo `model_trainer.py` implementa un pipeline completo para:
- Preprocesar imágenes
- Extraer características usando los detectores individuales
- Entrenar un modelo combinado basado en MobileNetV2
- Evaluar el rendimiento en conjuntos de prueba
- Guardar y cargar modelos entrenados

### 3. Sistema de Validación

El módulo `image_validator.py` proporciona:
- Validación de imágenes usando detectores individuales o el modelo entrenado
- Visualización de resultados con anotaciones en la imagen
- Generación de informes detallados en formato HTML
- Exportación de resultados en formato JSON

## Resultados de Evaluación

Los detectores fueron evaluados con un conjunto de imágenes de prueba, obteniendo los siguientes resultados:

| Detector | Exactitud | Precisión | Exhaustividad | F1 Score |
|----------|-----------|-----------|---------------|----------|
| Fondo Blanco | 0.82 | 1.00 | 0.67 | 0.80 |
| Personas | 0.73 | 0.00 | 0.00 | 0.00 |
| Cédula de Identidad | 1.00 | 1.00 | 1.00 | 1.00 |
| RUT Chileno | 0.64 | 1.00 | 0.20 | 0.33 |

### Análisis de Resultados

- **Detector de Cédulas**: Excelente rendimiento, detectando correctamente todas las cédulas en las imágenes de prueba.
- **Detector de Fondo Blanco**: Buen rendimiento, con alta precisión pero moderada exhaustividad.
- **Detector de RUT**: Rendimiento moderado, con alta precisión pero baja exhaustividad.
- **Detector de Personas**: Rendimiento deficiente con las imágenes sintéticas utilizadas. Requiere mejoras o imágenes reales para entrenamiento.

## Guía de Uso

### Requisitos

- Python 3.10 o superior
- OpenCV
- TensorFlow
- scikit-learn
- scikit-image
- pytesseract
- Tesseract OCR instalado en el sistema

### Instalación

```bash
# Instalar dependencias de Python
pip install opencv-python tensorflow scikit-learn scikit-image matplotlib pytesseract

# Instalar Tesseract OCR (en sistemas basados en Debian/Ubuntu)
sudo apt-get update && sudo apt-get install -y tesseract-ocr
```

### Uso Básico

#### 1. Validar una imagen

```python
from image_validator import ImageValidator

# Crear validador
validator = ImageValidator()

# Validar imagen
results = validator.validate_image("ruta/a/imagen.jpg")

# Mostrar resultados
for feature, present in results.items():
    if feature not in ['confidence', 'metadata']:
        print(f"{feature}: {'Sí' if present else 'No'} (Confianza: {results['confidence'][feature]:.2f})")

# Generar informe
report_path = validator.save_validation_report("ruta/a/imagen.jpg", results, "ruta/a/resultados")
print(f"Informe guardado en: {report_path}")
```

#### 2. Entrenar un modelo personalizado

```python
from model_trainer import ModelTrainer

# Crear entrenador
trainer = ModelTrainer(
    dataset_dir="ruta/a/dataset",
    model_dir="ruta/a/modelos"
)

# Entrenar modelo
history = trainer.train(epochs=50, batch_size=32)

# Evaluar modelo
metrics = trainer.evaluate()
print(metrics)
```

#### 3. Uso desde línea de comandos

El sistema incluye una interfaz de línea de comandos para validar imágenes:

```bash
python image_validator.py ruta/a/imagen.jpg --output_dir ruta/a/resultados
```

## Limitaciones y Mejoras Futuras

### Limitaciones Actuales

1. **Detector de Personas**: El rendimiento es deficiente con imágenes sintéticas. Se recomienda:
   - Utilizar imágenes reales de personas para entrenamiento
   - Implementar un detector basado en HOG + SVM o YOLO como alternativa

2. **Detector de RUT**: Baja exhaustividad en la detección. Se recomienda:
   - Mejorar el preprocesamiento de imágenes para OCR
   - Entrenar un modelo específico para localización de texto en documentos

3. **Conjunto de Datos**: El sistema actual utiliza un conjunto de datos sintético. Para aplicaciones reales:
   - Crear un conjunto de datos con imágenes reales
   - Implementar técnicas de aumento de datos

### Mejoras Propuestas

1. **Interfaz Gráfica**: Desarrollar una interfaz web o de escritorio para facilitar el uso.
2. **Procesamiento por Lotes**: Implementar funcionalidad para procesar múltiples imágenes en lote.
3. **Optimización de Rendimiento**: Mejorar la velocidad de procesamiento para aplicaciones en tiempo real.
4. **Integración con APIs**: Permitir el uso del sistema como un servicio web.
5. **Soporte para Más Características**: Ampliar el sistema para detectar otras características relevantes.

## Conclusiones

El sistema desarrollado proporciona una base sólida para la detección y validación de características específicas en imágenes. Los detectores de fondo blanco y cédulas de identidad muestran un rendimiento excelente, mientras que los detectores de RUT y personas requieren mejoras adicionales.

La arquitectura modular permite extender fácilmente el sistema para incluir nuevas características o mejorar los detectores existentes. El pipeline de entrenamiento integrado facilita la creación de modelos personalizados adaptados a necesidades específicas.

Para aplicaciones en producción, se recomienda entrenar los modelos con conjuntos de datos más grandes y representativos, así como implementar las mejoras propuestas para abordar las limitaciones actuales.
