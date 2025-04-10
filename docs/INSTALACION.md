# Guía de Instalación - Detector de Patrones Oscuros

Este documento proporciona instrucciones detalladas para instalar y configurar el sistema de detección de patrones oscuros en sitios web.

## Requisitos del Sistema

- Python 3.8 o superior
- Navegador Chromium, Chrome o Edge instalado (para Playwright)
- Aproximadamente 500 MB de espacio libre en disco para dependencias
- Conexión a Internet para analizar sitios web

## Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/dark-patterns-detector.git
cd dark-patterns-detector
```

### 2. Crear un Entorno Virtual (Recomendado)

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar el entorno virtual
# En Linux/macOS:
source venv/bin/activate
# En Windows:
venv\Scripts\activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Instalar Navegadores para Playwright

```bash
python -m playwright install
```

### 5. Instalar Dependencias del Sistema (Solo Linux)

En sistemas basados en Ubuntu/Debian, es posible que necesites instalar algunas dependencias adicionales para Playwright:

```bash
sudo apt-get update
sudo apt-get install -y libgstreamer1.0-dev libgtk-3-dev libatomic1 libxslt1-dev libwebp-dev libevent-dev libopus-dev
```

## Verificación de la Instalación

Para verificar que la instalación se ha completado correctamente, puedes ejecutar el script de prueba incluido:

```bash
python test_detector.py
```

Si todo está configurado correctamente, verás la salida del análisis de algunos sitios web de ejemplo.

## Estructura de Directorios

Después de la instalación, la estructura de directorios del proyecto debería verse así:

```
dark_patterns_detector/
├── data/                  # Datos de entrada y salida
│   ├── reports/           # Informes generados
│   ├── screenshots/       # Capturas de pantalla
│   └── test_sites.csv     # Sitios de prueba
├── docs/                  # Documentación
├── src/                   # Código fuente
│   ├── crawlers/          # Navegador automatizado
│   ├── detectors/         # Detectores de patrones
│   ├── reports/           # Generador de informes
│   ├── utils/             # Utilidades
│   └── web/               # Interfaz web
├── tests/                 # Pruebas
├── README.md              # Descripción general
└── requirements.txt       # Dependencias
```

## Solución de Problemas Comunes

### Error al instalar Playwright

Si encuentras errores al instalar Playwright, intenta:

```bash
pip install --upgrade pip
pip install playwright
python -m playwright install --with-deps
```

### Problemas con el Navegador

Si el navegador no se inicia correctamente:

1. Verifica que tienes instalado Chrome, Chromium o Edge
2. Intenta reinstalar los navegadores de Playwright:
   ```bash
   python -m playwright install --force
   ```

### Errores de Dependencias en Linux

Si encuentras errores relacionados con bibliotecas faltantes en Linux:

```bash
sudo apt-get update
sudo apt-get install -y libgstreamer1.0-dev libgtk-3-dev libatomic1 libxslt1-dev libwebp-dev libevent-dev libopus-dev
```

## Próximos Pasos

Una vez completada la instalación, consulta el archivo `USO.md` para obtener instrucciones sobre cómo utilizar el sistema.

---

Para obtener ayuda adicional o reportar problemas, por favor crea un issue en el repositorio del proyecto.
