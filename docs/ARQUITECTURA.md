# Documentación Técnica - Arquitectura del Sistema

Este documento describe la arquitectura técnica del Detector de Patrones Oscuros, explicando sus componentes principales, interacciones y decisiones de diseño.

## Visión General

El sistema está diseñado con una arquitectura modular que permite la detección automatizada de patrones oscuros en sitios web. La arquitectura se compone de cinco módulos principales:

1. **Sistema de Carga de URLs**: Gestiona la entrada de URLs desde diferentes fuentes
2. **Navegador Automatizado**: Controla la navegación web y captura de contenido
3. **Detectores de Patrones**: Analizan el contenido para identificar patrones oscuros
4. **Generador de Informes**: Procesa y formatea los resultados de la detección
5. **Interfaz Web**: Proporciona una interfaz de usuario para el sistema

## Diagrama de Arquitectura

```
+------------------+     +---------------------+     +-------------------+
| Sistema de Carga |---->| Navegador           |---->| Detectores de     |
| de URLs          |     | Automatizado        |     | Patrones Oscuros  |
+------------------+     +---------------------+     +-------------------+
                                                             |
                         +---------------------+             |
                         | Interfaz Web        |<------------+
                         |                     |             |
                         +---------------------+             |
                                   ^                         v
                                   |                +-------------------+
                                   +----------------|  Generador de     |
                                                    |  Informes         |
                                                    +-------------------+
```

## Componentes Principales

### 1. Sistema de Carga de URLs (`src/utils/url_loader.py`)

Este componente es responsable de cargar y validar URLs desde diferentes fuentes:

- **Características clave**:
  - Soporte para archivos CSV, JSON y TXT
  - Validación de formato de URL
  - Eliminación de duplicados
  - Gestión de cola de procesamiento

- **Clases principales**:
  - `URLLoader`: Carga URLs desde archivos
  - `URLValidator`: Valida el formato de las URLs
  - `URLQueue`: Gestiona la cola de procesamiento

### 2. Navegador Automatizado (`src/crawlers/web_crawler.py`)

Este componente controla un navegador headless para navegar y extraer información de sitios web:

- **Características clave**:
  - Navegación automatizada con Playwright
  - Captura de pantalla de páginas completas y elementos específicos
  - Extracción de estructura DOM
  - Simulación de interacciones de usuario

- **Clases principales**:
  - `WebCrawler`: Clase base para navegación web
  - `DarkPatternCrawler`: Especialización para detección de patrones oscuros

- **Decisiones técnicas**:
  - Se eligió Playwright sobre Selenium por su mejor rendimiento y capacidades modernas
  - El modo headless es configurable para permitir depuración visual
  - Se implementó manejo robusto de errores para sitios problemáticos

### 3. Detectores de Patrones Oscuros (`src/detectors/`)

Este componente contiene los algoritmos para detectar diferentes tipos de patrones oscuros:

- **Características clave**:
  - Arquitectura basada en plugins (cada detector es independiente)
  - Sistema de puntuación de confianza configurable
  - Capacidad para capturar evidencias específicas

- **Detectores implementados**:
  - `ConfirmshamingDetector`: Detecta textos que avergüenzan al usuario
  - `PreselectionDetector`: Identifica opciones preseleccionadas
  - `HiddenCostsDetector`: Encuentra cargos ocultos
  - `DifficultCancellationDetector`: Detecta procesos de cancelación complicados
  - `MisleadingAdsDetector`: Identifica publicidad engañosa
  - `FalseUrgencyDetector`: Detecta contadores falsos de urgencia/escasez
  - `ConfusingInterfaceDetector`: Encuentra interfaces confusas o engañosas

- **Clase base**:
  - `DarkPatternDetector`: Define la interfaz común y funcionalidades compartidas

### 4. Generador de Informes (`src/reports/report_generator.py`)

Este componente procesa los resultados de la detección y genera informes en diferentes formatos:

- **Características clave**:
  - Generación de informes en formatos JSON, CSV y HTML
  - Cálculo de puntuaciones de severidad
  - Inclusión de capturas de pantalla como evidencia
  - Sugerencias de mejora para cada patrón detectado

- **Clases principales**:
  - `ReportGenerator`: Genera informes individuales
  - `ReportManager`: Gestiona múltiples informes y genera resúmenes

### 5. Interfaz Web (`src/web/`)

Este componente proporciona una interfaz de usuario web para el sistema:

- **Características clave**:
  - Carga de URLs mediante archivos o entrada directa
  - Monitoreo de progreso en tiempo real
  - Visualización de resultados con gráficos
  - Exportación de informes

- **Tecnologías utilizadas**:
  - Backend: Flask
  - Frontend: HTML5, CSS3, JavaScript
  - Gráficos: Chart.js

- **Archivos principales**:
  - `app.py`: Aplicación Flask principal
  - `templates/`: Plantillas HTML
  - `static/`: Archivos CSS y JavaScript

## Flujo de Datos

1. El usuario proporciona URLs a través de la interfaz web o línea de comandos
2. El sistema de carga procesa y valida las URLs
3. El navegador automatizado visita cada URL y captura contenido
4. Los detectores de patrones analizan el contenido y generan detecciones
5. El generador de informes procesa las detecciones y crea informes
6. Los resultados se muestran al usuario a través de la interfaz web o archivos

## Patrones de Diseño Utilizados

- **Patrón Estrategia**: Utilizado en los detectores de patrones para permitir diferentes algoritmos de detección
- **Patrón Fachada**: Implementado en la clase principal para simplificar el uso del sistema
- **Patrón Observador**: Utilizado para notificar progreso en tiempo real
- **Patrón Singleton**: Aplicado en componentes como el gestor de configuración
- **Patrón Decorador**: Usado para añadir funcionalidades como logging y medición de rendimiento

## Consideraciones de Rendimiento

- **Paralelismo**: El sistema puede procesar múltiples URLs en paralelo
- **Caché**: Se implementa caché de resultados para evitar análisis repetidos
- **Optimización de recursos**: Control de uso de memoria en capturas de pantalla
- **Tiempos de espera adaptativos**: Ajuste dinámico según la respuesta del sitio

## Seguridad

- **Aislamiento**: El navegador se ejecuta en modo aislado
- **Validación de entrada**: Todas las entradas del usuario son validadas
- **Manejo de errores**: Captura y registro de excepciones
- **Limitación de recursos**: Prevención de ataques DoS

## Extensibilidad

El sistema está diseñado para ser fácilmente extensible:

- **Nuevos detectores**: Simplemente crear una nueva clase que herede de `DarkPatternDetector`
- **Formatos de informe adicionales**: Extender `ReportGenerator` con nuevos métodos
- **Soporte para más navegadores**: Configurar Playwright para usar diferentes navegadores
- **Integración con otras herramientas**: API bien definida para integración externa

## Limitaciones Conocidas

- **Sitios con protección anti-bot**: Algunos sitios pueden bloquear el navegador automatizado
- **Contenido dinámico complejo**: JavaScript avanzado puede ser difícil de analizar
- **Idiomas**: Actualmente optimizado para español e inglés
- **Rendimiento**: El análisis profundo puede ser lento en sitios muy grandes

## Futuras Mejoras

- Implementación de aprendizaje automático para mejorar la detección
- Soporte para más idiomas
- Análisis de flujos completos (no solo páginas individuales)
- API REST para integración con otros sistemas
- Soporte para aplicaciones móviles

---

Para más información sobre la implementación específica de cada componente, consulte los comentarios en el código fuente.
