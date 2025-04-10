# Sistema de Detección de Patrones Oscuros

Este proyecto implementa un sistema automatizado para detectar patrones oscuros (dark patterns) en sitios web. El sistema navega automáticamente por sitios web, identifica patrones oscuros comunes y genera informes detallados.

## Características

- Carga de URLs desde archivos CSV, JSON o TXT
- Navegación automatizada de sitios web
- Detección de múltiples tipos de patrones oscuros:
  - Confirmshaming
  - Preselección de opciones
  - Cargos ocultos
  - Suscripciones difíciles de cancelar
  - Publicidad engañosa
  - Falsos contadores de urgencia o escasez
  - Interfaces confusas o botones engañosos
- Generación de informes detallados con evidencias
- Interfaz web para visualización de resultados

## Estructura del Proyecto

```
dark_patterns_detector/
├── src/                    # Código fuente
│   ├── crawlers/           # Módulos de navegación web
│   ├── detectors/          # Detectores de patrones oscuros
│   ├── utils/              # Utilidades y herramientas
│   ├── reports/            # Generadores de informes
│   └── web/                # Interfaz web
├── data/                   # Datos de entrada y salida
├── tests/                  # Pruebas unitarias y de integración
└── docs/                   # Documentación
```

## Requisitos

- Python 3.8+
- Navegador web (Chrome/Firefox)
- Dependencias listadas en requirements.txt

## Instalación

1. Clonar el repositorio
2. Instalar dependencias: `pip install -r requirements.txt`
3. Configurar el entorno según la documentación

## Uso

1. Preparar un archivo con lista de URLs (CSV, JSON o TXT)
2. Ejecutar el sistema: `python -m src.main --input urls.csv`
3. Acceder a la interfaz web: `http://localhost:5000`
4. Ver y exportar los informes generados

## Licencia

Este proyecto está disponible bajo la licencia MIT.
