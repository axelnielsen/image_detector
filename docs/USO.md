# Manual de Uso - Detector de Patrones Oscuros

Este documento explica cómo utilizar el sistema de detección de patrones oscuros para analizar sitios web y generar informes detallados.

## Introducción

El Detector de Patrones Oscuros es una herramienta diseñada para identificar automáticamente prácticas manipulativas (dark patterns) en sitios web. La herramienta puede detectar:

- **Confirmshaming**: Textos que avergüenzan al usuario por rechazar una oferta
- **Preselección de opciones**: Opciones marcadas por defecto que benefician al sitio
- **Cargos ocultos**: Costos adicionales revelados tarde en el proceso
- **Suscripciones difíciles de cancelar**: Procesos asimétricos de suscripción/cancelación
- **Publicidad engañosa**: Anuncios camuflados como contenido o funcionalidades
- **Falsos contadores de urgencia**: Creación artificial de sensación de urgencia o escasez
- **Interfaces confusas**: Diseños que confunden al usuario para realizar acciones no deseadas

## Modos de Uso

El sistema puede utilizarse de tres formas diferentes:

1. **Línea de comandos**: Para análisis rápidos o automatizados
2. **Interfaz web**: Para uso interactivo con visualización de resultados
3. **Como biblioteca**: Para integrar en otros proyectos Python

## Uso desde Línea de Comandos

### Analizar un Sitio Web Individual

```bash
python -m src.main --url https://www.ejemplo.com
```

### Analizar Múltiples Sitios desde un Archivo

```bash
# Desde un archivo CSV
python -m src.main --file data/mis_sitios.csv

# Desde un archivo JSON
python -m src.main --file data/mis_sitios.json

# Desde un archivo de texto (una URL por línea)
python -m src.main --file data/mis_sitios.txt
```

### Opciones Adicionales

```bash
# Especificar directorio de salida para informes
python -m src.main --url https://www.ejemplo.com --output-dir mis_informes

# Ejecutar en modo visible (no headless)
python -m src.main --url https://www.ejemplo.com --no-headless

# Generar solo informes específicos
python -m src.main --url https://www.ejemplo.com --formats json,html

# Establecer nivel de confianza mínimo (0.0-1.0)
python -m src.main --url https://www.ejemplo.com --confidence 0.7

# Ayuda completa
python -m src.main --help
```

## Uso de la Interfaz Web

### Iniciar la Interfaz Web

```bash
python -m src.web.app
```

Por defecto, la interfaz web estará disponible en `http://localhost:5000`.

### Funcionalidades de la Interfaz Web

1. **Página de inicio**:
   - Cargar archivo de URLs (CSV, JSON, TXT)
   - Ingresar URLs directamente en un campo de texto
   - Iniciar análisis

2. **Página de estado de tarea**:
   - Ver progreso del análisis en tiempo real
   - Visualizar resultados cuando el análisis se completa
   - Exportar informes en diferentes formatos

3. **Dashboard**:
   - Ver estadísticas de todos los análisis realizados
   - Visualizar gráficos de distribución de patrones
   - Información detallada sobre cada tipo de patrón

## Uso como Biblioteca

Puedes integrar el detector en tus propios proyectos Python:

```python
from src.utils.url_loader import URLLoader
from src.crawlers.web_crawler import DarkPatternCrawler
from src.detectors.confirmshaming_detector import ConfirmshamingDetector
# Importar otros detectores según sea necesario
from src.reports.report_generator import ReportGenerator

# Configurar crawler
with DarkPatternCrawler(headless=True) as crawler:
    # Navegar a una URL
    result = crawler.analyze_page("https://www.ejemplo.com")
    
    if result["success"]:
        # Obtener contenido y estructura DOM
        page_content = crawler.get_page_content()
        dom_structure = result.get("dom_structure", {})
        
        # Configurar detector
        detector = ConfirmshamingDetector()
        
        # Detectar patrones
        detections = detector.detect(
            page_content=page_content,
            dom_structure=dom_structure,
            screenshot_path=result["screenshots"]["full"],
            url="https://www.ejemplo.com"
        )
        
        # Generar informe
        if detections:
            report_generator = ReportGenerator("informes")
            report = report_generator.generate_report(
                url="https://www.ejemplo.com",
                detections=detections,
                screenshots=result["screenshots"],
                metadata={"title": result.get("title", "Sin título")}
            )
            
            # Guardar informe
            json_path = report_generator.save_report_json(report)
            print(f"Informe guardado en: {json_path}")
```

## Formato de los Archivos de Entrada

### Archivo CSV

```csv
url,category,notes
https://www.ejemplo1.com,ecommerce,Sitio de comercio electrónico
https://www.ejemplo2.com,travel,Sitio de viajes
```

### Archivo JSON

```json
[
  {
    "url": "https://www.ejemplo1.com",
    "category": "ecommerce",
    "notes": "Sitio de comercio electrónico"
  },
  {
    "url": "https://www.ejemplo2.com",
    "category": "travel",
    "notes": "Sitio de viajes"
  }
]
```

### Archivo TXT

```
https://www.ejemplo1.com
https://www.ejemplo2.com
```

## Interpretación de Resultados

Los informes generados incluyen:

- **Tipo de patrón**: Categoría del patrón oscuro detectado
- **Confianza**: Nivel de confianza en la detección (0.0-1.0)
- **Ubicación**: Dónde se encontró el patrón en la página
- **Evidencia**: Datos que respaldan la detección
- **Captura de pantalla**: Imagen de la página con el patrón marcado

## Solución de Problemas

### El análisis falla para ciertos sitios

Algunos sitios web implementan mecanismos anti-bot que pueden interferir con el análisis. Prueba estas soluciones:

1. Utiliza la opción `--no-headless` para ejecutar en modo visible
2. Aumenta el tiempo de espera con `--timeout 60000` (en milisegundos)
3. Añade retrasos entre acciones con `--delay 2` (en segundos)

### Falsos positivos o negativos

Si encuentras detecciones incorrectas:

1. Ajusta el nivel de confianza con `--confidence` (valores más altos reducen falsos positivos)
2. Verifica los informes detallados para entender por qué se produjo la detección
3. Considera contribuir al proyecto reportando estos casos para mejorar los detectores

---

Para más información, consulta la documentación técnica en el directorio `docs/` o visita el repositorio del proyecto.
