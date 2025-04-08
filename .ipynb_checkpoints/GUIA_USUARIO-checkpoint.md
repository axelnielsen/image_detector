"""
Guía de Usuario - Sistema de Detección y Validación de Imágenes

Este documento proporciona instrucciones detalladas para utilizar el sistema
de detección y validación de imágenes desarrollado.

Autor: Sistema de Detección de Imágenes
Fecha: Abril 2025
"""

# Guía de Usuario - Sistema de Detección y Validación de Imágenes

## Introducción

El Sistema de Detección y Validación de Imágenes es una herramienta diseñada para identificar características específicas en imágenes, como:

- Presencia de fondo blanco
- Detección de personas
- Identificación de cédulas de identidad
- Reconocimiento de RUT chileno

Esta guía le ayudará a instalar, configurar y utilizar el sistema para sus necesidades específicas.

## Requisitos del Sistema

### Requisitos de Hardware
- Procesador: 2 GHz o superior
- RAM: 4 GB mínimo (8 GB recomendado)
- Espacio en disco: 1 GB mínimo para la instalación

### Requisitos de Software
- Sistema Operativo: Windows 10/11, macOS 10.14+, o Linux (Ubuntu 20.04+)
- Python 3.10 o superior
- Bibliotecas de Python (instaladas automáticamente):
  - OpenCV
  - TensorFlow
  - scikit-learn
  - scikit-image
  - pytesseract
- Tesseract OCR (instalación separada)

## Instalación

### Paso 1: Instalar Python
Asegúrese de tener Python 3.10 o superior instalado. Puede descargarlo desde [python.org](https://www.python.org/downloads/).

### Paso 2: Instalar Tesseract OCR
- **Windows**: Descargue el instalador desde [GitHub](https://github.com/UB-Mannheim/tesseract/wiki) e instálelo.
- **macOS**: Use Homebrew: `brew install tesseract`
- **Linux**: Use apt: `sudo apt-get install tesseract-ocr`

### Paso 3: Instalar el Sistema de Detección
1. Descomprima el archivo ZIP del sistema en la ubicación deseada
2. Abra una terminal o línea de comandos
3. Navegue al directorio donde descomprimió el sistema
4. Instale las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso Básico

### Validar una Imagen Individual

Para validar una imagen y detectar sus características:

1. Abra una terminal o línea de comandos
2. Navegue al directorio del sistema
3. Ejecute el siguiente comando:
   ```bash
   python image_validator.py ruta/a/su/imagen.jpg --output_dir resultados
   ```
4. Los resultados se guardarán en la carpeta `resultados` con:
   - Imagen con anotaciones visuales
   - Archivo JSON con los resultados detallados
   - Informe HTML para visualización

### Interpretar los Resultados

El sistema genera tres tipos de archivos de resultados:

1. **Imagen con anotaciones**: Muestra visualmente las características detectadas
2. **Archivo JSON**: Contiene los resultados detallados en formato estructurado
3. **Informe HTML**: Presenta los resultados en un formato fácil de leer

El informe HTML incluye:
- Resumen de características detectadas
- Nivel de confianza para cada detección
- Imágenes comparativas (original y con anotaciones)
- Detalles adicionales como porcentaje de fondo blanco o RUT detectado

## Uso Avanzado

### Entrenar un Modelo Personalizado

Si desea entrenar el sistema con sus propias imágenes:

1. Organice sus imágenes en la estructura de directorios requerida:
   ```
   dataset/
   ├── train/
   │   ├── fondo_blanco/
   │   │   ├── positive/
   │   │   └── negative/
   │   ├── personas/
   │   │   ├── positive/
   │   │   └── negative/
   │   ├── cedula_identidad/
   │   │   ├── positive/
   │   │   └── negative/
   │   └── rut_chileno/
   │       ├── positive/
   │       └── negative/
   └── test/
       ├── [estructura similar]
   ```

2. Ejecute el script de entrenamiento:
   ```bash
   python train_model.py --dataset_dir ruta/a/dataset --model_dir modelos --epochs 50
   ```

3. El modelo entrenado se guardará en el directorio `modelos`

### Procesamiento por Lotes

Para procesar múltiples imágenes a la vez:

1. Coloque todas las imágenes en un directorio
2. Ejecute:
   ```bash
   python batch_validator.py --input_dir ruta/a/imagenes --output_dir resultados
   ```

### Ajustar Parámetros de Detección

Puede modificar los umbrales de detección editando el archivo `config.json`:

```json
{
  "fondo_blanco": {
    "threshold": 0.85,
    "white_threshold": 230
  },
  "persona": {
    "confidence_threshold": 0.5
  },
  "cedula": {
    "min_area_ratio": 0.1,
    "max_area_ratio": 0.9,
    "aspect_ratio_range": [1.4, 1.7]
  },
  "rut": {
    "confidence_threshold": 0.5
  }
}
```

## Solución de Problemas

### Problemas Comunes

1. **Error: "No se pudo encontrar Tesseract OCR"**
   - Asegúrese de que Tesseract OCR esté instalado correctamente
   - Verifique que la ruta a Tesseract esté en la variable PATH del sistema

2. **Detección de personas no funciona correctamente**
   - El detector de personas funciona mejor con fotografías reales
   - Las siluetas o dibujos pueden no ser reconocidos correctamente

3. **Bajo rendimiento en la detección de RUT**
   - Asegúrese de que la imagen tenga buena resolución
   - El texto debe ser claramente legible
   - Pruebe con diferentes ángulos o iluminación

### Contacto y Soporte

Si encuentra problemas o tiene preguntas, puede:
- Consultar la documentación completa en el archivo README.md
- Revisar los ejemplos en la carpeta `examples`
- Contactar al equipo de soporte

## Ejemplos de Uso

### Ejemplo 1: Validar una cédula de identidad
```bash
python image_validator.py ejemplos/cedula.jpg --output_dir resultados
```

### Ejemplo 2: Entrenar con un conjunto de datos personalizado
```bash
python train_model.py --dataset_dir mi_dataset --model_dir mis_modelos --epochs 30
```

### Ejemplo 3: Usar el sistema desde Python
```python
from image_validator import ImageValidator

validator = ImageValidator()
results = validator.validate_image("mi_imagen.jpg")
print(results)
```

## Conclusión

Este sistema proporciona una solución robusta para la detección de características específicas en imágenes. Con la configuración adecuada y siguiendo las instrucciones de esta guía, podrá obtener resultados precisos para sus necesidades de validación de imágenes.
