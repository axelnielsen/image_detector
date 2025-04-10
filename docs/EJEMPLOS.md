# Ejemplos de Uso - Detector de Patrones Oscuros

Este documento proporciona ejemplos prácticos de uso del sistema de detección de patrones oscuros en diferentes escenarios.

## Ejemplo 1: Análisis Básico de un Sitio Web

Este ejemplo muestra cómo analizar un único sitio web desde la línea de comandos:

```bash
python -m src.main --url https://www.ejemplo.com
```

**Resultado esperado:**
- Se abrirá un navegador headless
- Se navegará a la URL especificada
- Se analizará la página en busca de patrones oscuros
- Se generarán informes en la carpeta `data/reports/`

## Ejemplo 2: Análisis de Múltiples Sitios

Este ejemplo muestra cómo analizar varios sitios web desde un archivo CSV:

**Archivo: sitios_ecommerce.csv**
```csv
url,category,notes
https://www.tienda1.com,ecommerce,Tienda de electrónica
https://www.tienda2.com,ecommerce,Tienda de ropa
https://www.tienda3.com,ecommerce,Tienda de alimentos
```

**Comando:**
```bash
python -m src.main --file data/sitios_ecommerce.csv --output-dir informes_ecommerce
```

**Resultado esperado:**
- Se analizarán los tres sitios secuencialmente
- Se generarán informes individuales para cada sitio
- Se creará un informe resumen con estadísticas comparativas
- Todos los informes se guardarán en la carpeta `informes_ecommerce/`

## Ejemplo 3: Uso de la Interfaz Web

Este ejemplo muestra cómo utilizar la interfaz web para análisis interactivo:

```bash
# Iniciar la interfaz web
python -m src.web.app
```

**Pasos:**
1. Abrir un navegador y visitar `http://localhost:5000`
2. En la página principal, escribir las siguientes URLs en el campo de texto:
   ```
   https://www.ejemplo1.com
   https://www.ejemplo2.com
   ```
3. Hacer clic en "Analizar URLs"
4. Observar el progreso del análisis en tiempo real
5. Una vez completado, explorar los resultados y descargar los informes

## Ejemplo 4: Integración en Script Personalizado

Este ejemplo muestra cómo integrar el detector en un script Python personalizado:

```python
# analisis_personalizado.py
import os
import sys
from datetime import datetime

# Añadir el directorio raíz al path
sys.path.append('.')

from src.utils.url_loader import URLLoader
from src.crawlers.web_crawler import DarkPatternCrawler
from src.detectors.confirmshaming_detector import ConfirmshamingDetector
from src.detectors.preselection_detector import PreselectionDetector
from src.detectors.hidden_costs_detector import HiddenCostsDetector
from src.reports.report_generator import ReportGenerator

# Lista de URLs a analizar
urls = [
    "https://www.ejemplo1.com",
    "https://www.ejemplo2.com",
    "https://www.ejemplo3.com"
]

# Configurar detectores
detectors = [
    ConfirmshamingDetector(),
    PreselectionDetector(),
    HiddenCostsDetector()
]

# Configurar generador de informes
report_dir = f"informes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(report_dir, exist_ok=True)
report_generator = ReportGenerator(report_dir)

# Analizar cada URL
results = {}
with DarkPatternCrawler(headless=True) as crawler:
    for url in urls:
        print(f"Analizando: {url}")
        
        # Navegar a la URL
        result = crawler.analyze_page(url)
        
        if not result["success"]:
            print(f"Error al analizar {url}: {result.get('error', 'Error desconocido')}")
            continue
        
        # Obtener contenido y estructura DOM
        page_content = crawler.get_page_content()
        dom_structure = result.get("dom_structure", {})
        
        # Ejecutar detectores
        all_detections = []
        for detector in detectors:
            detections = detector.detect(
                page_content=page_content,
                dom_structure=dom_structure,
                screenshot_path=result["screenshots"]["full"],
                url=url
            )
            all_detections.extend(detections)
        
        # Generar informe si se encontraron patrones
        if all_detections:
            report = report_generator.generate_report(
                url=url,
                detections=all_detections,
                screenshots=result["screenshots"],
                metadata={"title": result.get("title", "Sin título")}
            )
            
            # Guardar informe
            json_path = report_generator.save_report_json(report)
            html_path = report_generator.generate_html_report(report)
            
            print(f"Informe JSON: {json_path}")
            print(f"Informe HTML: {html_path}")
            
            # Guardar resultado
            results[url] = {
                "success": True,
                "detections": len(all_detections),
                "patterns": list(set(d["pattern_type"] for d in all_detections)),
                "reports": {
                    "json": json_path,
                    "html": html_path
                }
            }
        else:
            print(f"No se encontraron patrones oscuros en {url}")
            results[url] = {
                "success": True,
                "detections": 0,
                "patterns": [],
                "reports": {}
            }

# Imprimir resumen
print("\nResumen de análisis:")
for url, result in results.items():
    if result["detections"] > 0:
        patterns = ", ".join(result["patterns"])
        print(f"- {url}: {result['detections']} patrones detectados ({patterns})")
    else:
        print(f"- {url}: Sin patrones detectados")
```

**Ejecución:**
```bash
python analisis_personalizado.py
```

## Ejemplo 5: Análisis con Configuración Avanzada

Este ejemplo muestra cómo utilizar opciones avanzadas para personalizar el análisis:

```bash
python -m src.main \
  --url https://www.ejemplo.com \
  --no-headless \
  --timeout 60000 \
  --delay 2 \
  --confidence 0.8 \
  --depth 3 \
  --formats json,html,csv \
  --output-dir informes_detallados \
  --screenshots \
  --verbose
```

Esta configuración:
- Ejecuta el navegador en modo visible (`--no-headless`)
- Establece un tiempo de espera de 60 segundos (`--timeout 60000`)
- Añade un retraso de 2 segundos entre acciones (`--delay 2`)
- Establece un umbral de confianza del 80% (`--confidence 0.8`)
- Analiza hasta 3 niveles de profundidad en el sitio (`--depth 3`)
- Genera informes en formatos JSON, HTML y CSV (`--formats json,html,csv`)
- Guarda los informes en el directorio `informes_detallados/`
- Guarda capturas de pantalla de cada patrón detectado (`--screenshots`)
- Muestra información detallada durante el análisis (`--verbose`)

## Ejemplo 6: Análisis Programado con Cron

Este ejemplo muestra cómo configurar un análisis automático periódico usando cron:

**Script: analisis_programado.sh**
```bash
#!/bin/bash

# Configurar variables
FECHA=$(date +%Y%m%d)
DIRECTORIO_PROYECTO="/ruta/a/dark_patterns_detector"
DIRECTORIO_SALIDA="/ruta/a/informes/$FECHA"
ARCHIVO_URLS="$DIRECTORIO_PROYECTO/data/sitios_monitoreo.csv"
ARCHIVO_LOG="$DIRECTORIO_PROYECTO/logs/analisis_$FECHA.log"

# Crear directorio de salida
mkdir -p "$DIRECTORIO_SALIDA"

# Activar entorno virtual
source "$DIRECTORIO_PROYECTO/venv/bin/activate"

# Ejecutar análisis
cd "$DIRECTORIO_PROYECTO"
python -m src.main \
  --file "$ARCHIVO_URLS" \
  --output-dir "$DIRECTORIO_SALIDA" \
  --formats json,html,csv \
  --verbose > "$ARCHIVO_LOG" 2>&1

# Enviar notificación por correo
if [ -f "$DIRECTORIO_SALIDA/summary_report.csv" ]; then
  echo "Análisis de patrones oscuros completado. Ver resultados en $DIRECTORIO_SALIDA" | \
  mail -s "Informe diario de patrones oscuros - $FECHA" usuario@ejemplo.com
fi
```

**Configuración de cron:**
```
# Ejecutar análisis todos los días a las 3:00 AM
0 3 * * * /ruta/a/dark_patterns_detector/analisis_programado.sh
```

## Ejemplo 7: Exportación de Resultados a Base de Datos

Este ejemplo muestra cómo exportar los resultados a una base de datos MySQL:

```python
# exportar_a_bd.py
import sys
import json
import mysql.connector
from datetime import datetime
from pathlib import Path

# Configuración de la base de datos
DB_CONFIG = {
    "host": "localhost",
    "user": "usuario",
    "password": "contraseña",
    "database": "dark_patterns_db"
}

# Directorio de informes
REPORTS_DIR = "data/reports"

# Conectar a la base de datos
try:
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    print("Conexión a la base de datos establecida")
except Exception as e:
    print(f"Error al conectar a la base de datos: {e}")
    sys.exit(1)

# Buscar archivos JSON de informes
report_files = list(Path(REPORTS_DIR).glob("*.json"))
print(f"Se encontraron {len(report_files)} informes para procesar")

# Procesar cada informe
for report_file in report_files:
    try:
        # Cargar informe JSON
        with open(report_file, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        # Insertar datos del sitio
        cursor.execute(
            "INSERT INTO sites (url, title, analysis_date) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE title=%s, analysis_date=%s",
            (
                report["url"],
                report["metadata"].get("title", "Sin título"),
                datetime.now(),
                report["metadata"].get("title", "Sin título"),
                datetime.now()
            )
        )
        site_id = cursor.lastrowid
        
        # Insertar detecciones
        for detection in report.get("detections", []):
            cursor.execute(
                "INSERT INTO detections (site_id, pattern_type, confidence, location, evidence_type) VALUES (%s, %s, %s, %s, %s)",
                (
                    site_id,
                    detection["pattern_type"],
                    detection["confidence"],
                    detection.get("location", ""),
                    detection.get("evidence_type", "")
                )
            )
        
        conn.commit()
        print(f"Informe {report_file.name} procesado correctamente")
        
    except Exception as e:
        print(f"Error al procesar {report_file}: {e}")
        conn.rollback()

# Cerrar conexión
cursor.close()
conn.close()
print("Exportación completada")
```

**Ejecución:**
```bash
python exportar_a_bd.py
```

---

Estos ejemplos ilustran diferentes formas de utilizar el sistema de detección de patrones oscuros. Para más información sobre opciones específicas, consulte el manual de uso completo en `USO.md`.
