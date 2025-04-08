# Estructura del Conjunto de Datos

## Organización General

```
dataset/
├── train/                  # Datos de entrenamiento (80% del total)
│   ├── fondo_blanco/       # Imágenes para detección de fondo blanco
│   │   ├── positive/       # Ejemplos con fondo blanco
│   │   └── negative/       # Ejemplos sin fondo blanco
│   ├── personas/           # Imágenes para detección de personas
│   │   ├── positive/       # Ejemplos con personas
│   │   └── negative/       # Ejemplos sin personas
│   ├── cedula_identidad/   # Imágenes para detección de cédulas
│   │   ├── positive/       # Ejemplos de cédulas de identidad
│   │   └── negative/       # Ejemplos de otros documentos
│   └── rut_chileno/        # Imágenes para detección de RUT
│       ├── positive/       # Ejemplos con RUT visible
│       └── negative/       # Ejemplos sin RUT visible
└── test/                   # Datos de prueba (20% del total)
    ├── fondo_blanco/       # Estructura similar a train
    │   ├── positive/
    │   └── negative/
    ├── personas/
    │   ├── positive/
    │   └── negative/
    ├── cedula_identidad/
    │   ├── positive/
    │   └── negative/
    └── rut_chileno/
        ├── positive/
        └── negative/
```

## Formato de Etiquetas

Para cada característica, se utilizará un formato de etiquetas JSON que permitirá almacenar información detallada sobre cada imagen:

```json
{
  "filename": "imagen_001.jpg",
  "path": "dataset/train/fondo_blanco/positive/imagen_001.jpg",
  "features": {
    "fondo_blanco": {
      "present": true,
      "confidence": 0.95,
      "metadata": {
        "porcentaje_blanco": 92.3
      }
    },
    "persona": {
      "present": false,
      "confidence": 0.87
    },
    "cedula_identidad": {
      "present": false,
      "confidence": 0.98
    },
    "rut_chileno": {
      "present": false,
      "confidence": 0.99
    }
  },
  "annotations": {
    "bounding_boxes": []
  }
}
```

Para imágenes con personas o documentos, se incluirán bounding boxes:

```json
{
  "filename": "imagen_002.jpg",
  "path": "dataset/train/personas/positive/imagen_002.jpg",
  "features": {
    "fondo_blanco": {
      "present": true,
      "confidence": 0.89,
      "metadata": {
        "porcentaje_blanco": 85.7
      }
    },
    "persona": {
      "present": true,
      "confidence": 0.96
    },
    "cedula_identidad": {
      "present": false,
      "confidence": 0.12
    },
    "rut_chileno": {
      "present": false,
      "confidence": 0.05
    }
  },
  "annotations": {
    "bounding_boxes": [
      {
        "label": "persona",
        "coordinates": [120, 80, 450, 550]
      }
    ]
  }
}
```

## Archivos de Metadatos

Se crearán los siguientes archivos de metadatos:

1. `dataset_stats.json`: Estadísticas generales del conjunto de datos
2. `train_manifest.json`: Lista de todas las imágenes de entrenamiento con sus etiquetas
3. `test_manifest.json`: Lista de todas las imágenes de prueba con sus etiquetas

## Scripts de Procesamiento

Se implementarán los siguientes scripts para gestionar el conjunto de datos:

1. `dataset_generator.py`: Para generar datos sintéticos o aumentar datos existentes
2. `dataset_splitter.py`: Para dividir datos entre entrenamiento y prueba
3. `label_generator.py`: Para generar archivos de etiquetas automáticamente
4. `dataset_visualizer.py`: Para visualizar ejemplos del conjunto de datos

Esta estructura permitirá una gestión eficiente del conjunto de datos y facilitará el entrenamiento y evaluación de los modelos para cada característica a detectar.
