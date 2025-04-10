"""
Módulo principal para la interfaz web del detector de patrones oscuros.
Implementa un servidor web con Flask para cargar URLs, ejecutar análisis y visualizar resultados.
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import os
import json
import uuid
import threading
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Añadir el directorio raíz al path para importar módulos
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.url_loader import URLLoader, URLQueue
from src.crawlers.web_crawler import DarkPatternCrawler
from src.detectors.base_detector import DarkPatternDetector
from src.detectors.confirmshaming_detector import ConfirmshamingDetector
from src.detectors.preselection_detector import PreselectionDetector
from src.detectors.hidden_costs_detector import HiddenCostsDetector
from src.detectors.difficult_cancellation_detector import DifficultCancellationDetector
from src.detectors.misleading_ads_detector import MisleadingAdsDetector
from src.detectors.false_urgency_detector import FalseUrgencyDetector
from src.detectors.confusing_interface_detector import ConfusingInterfaceDetector
from src.reports.report_generator import ReportGenerator, ReportManager

# Crear aplicación Flask
app = Flask(__name__)

# Configuración
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_FOLDER = DATA_DIR / "uploads"
REPORTS_FOLDER = DATA_DIR / "reports"
SCREENSHOTS_FOLDER = DATA_DIR / "screenshots"
EVIDENCE_FOLDER = DATA_DIR / "evidence"

# Crear directorios si no existen
UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)
REPORTS_FOLDER.mkdir(exist_ok=True, parents=True)
SCREENSHOTS_FOLDER.mkdir(exist_ok=True, parents=True)
EVIDENCE_FOLDER.mkdir(exist_ok=True, parents=True)

# Configurar Flask
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Estado global
analysis_tasks = {}
url_queue = URLQueue()


class AnalysisTask:
    """Clase para gestionar tareas de análisis."""
    
    def __init__(self, task_id: str, urls: List[str]):
        """
        Inicializa una tarea de análisis.
        
        Args:
            task_id: Identificador único de la tarea
            urls: Lista de URLs a analizar
        """
        self.task_id = task_id
        self.urls = urls
        self.status = "pending"  # pending, running, completed, failed
        self.progress = 0
        self.total_urls = len(urls)
        self.results = {}
        self.reports = {}
        self.start_time = None
        self.end_time = None
        self.error = None
    
    def start(self):
        """Inicia la tarea de análisis."""
        self.status = "running"
        self.start_time = datetime.now()
        
        # Crear hilo para ejecutar el análisis
        thread = threading.Thread(target=self._run_analysis)
        thread.daemon = True
        thread.start()
    
    def _run_analysis(self):
        """Ejecuta el análisis en segundo plano."""
        try:
            # Inicializar detectores
            detectors = [
                ConfirmshamingDetector(),
                PreselectionDetector(),
                HiddenCostsDetector(),
                DifficultCancellationDetector(),
                MisleadingAdsDetector(),
                FalseUrgencyDetector(),
                ConfusingInterfaceDetector()
            ]
            
            # Inicializar generador de informes
            report_generator = ReportGenerator(str(REPORTS_FOLDER))
            
            # Analizar cada URL
            for i, url in enumerate(self.urls):
                try:
                    # Actualizar progreso
                    self.progress = (i / self.total_urls) * 100
                    
                    # Analizar URL
                    result = self._analyze_url(url, detectors)
                    
                    # Generar informe
                    if result["success"]:
                        report = report_generator.generate_report(
                            url=url,
                            detections=result["detections"],
                            screenshots=result["screenshots"],
                            metadata={"title": result["title"]}
                        )
                        
                        # Guardar informe en diferentes formatos
                        json_path = report_generator.save_report_json(report)
                        csv_path = report_generator.save_report_csv(report)
                        html_path = report_generator.generate_html_report(report)
                        
                        self.reports[url] = {
                            "json": json_path,
                            "csv": csv_path,
                            "html": html_path
                        }
                    
                    # Guardar resultado
                    self.results[url] = result
                
                except Exception as e:
                    # Registrar error para esta URL
                    self.results[url] = {
                        "url": url,
                        "success": False,
                        "error": str(e)
                    }
            
            # Completar tarea
            self.status = "completed"
            self.progress = 100
            self.end_time = datetime.now()
        
        except Exception as e:
            # Registrar error global
            self.status = "failed"
            self.error = str(e)
            self.end_time = datetime.now()
    
    def _analyze_url(self, url: str, detectors: List[DarkPatternDetector]) -> Dict[str, Any]:
        """
        Analiza una URL en busca de patrones oscuros.
        
        Args:
            url: URL a analizar
            detectors: Lista de detectores a utilizar
            
        Returns:
            Dict[str, Any]: Resultado del análisis
        """
        # Inicializar crawler
        with DarkPatternCrawler(headless=True, screenshots_dir=str(SCREENSHOTS_FOLDER)) as crawler:
            # Navegar a la URL
            result = crawler.analyze_page(url)
            
            if not result["success"]:
                return result
            
            # Obtener contenido de la página
            page_content = crawler.get_page_content()
            dom_structure = crawler.get_dom_structure()
            
            # Detectar patrones oscuros
            all_detections = []
            
            for detector in detectors:
                detections = detector.detect(
                    page_content=page_content,
                    dom_structure=dom_structure,
                    screenshot_path=result["screenshots"]["full"],
                    url=url
                )
                
                all_detections.extend(detections)
            
            # Añadir detecciones al resultado
            result["detections"] = all_detections
            
            return result
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual de la tarea.
        
        Returns:
            Dict[str, Any]: Estado de la tarea
        """
        status_data = {
            "task_id": self.task_id,
            "status": self.status,
            "progress": self.progress,
            "total_urls": self.total_urls,
            "processed_urls": len(self.results),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error": self.error
        }
        
        # Añadir resumen de resultados si está completado
        if self.status == "completed":
            status_data["summary"] = {
                "total_success": sum(1 for r in self.results.values() if r.get("success", False)),
                "total_failed": sum(1 for r in self.results.values() if not r.get("success", False)),
                "total_detections": sum(len(r.get("detections", [])) for r in self.results.values() if r.get("success", False))
            }
        
        return status_data


# Rutas de la aplicación
@app.route('/')
def index():
    """Página principal."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Maneja la carga de archivos con URLs."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        # Guardar archivo
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Cargar URLs
        try:
            loader = URLLoader(file_path)
            urls = loader.load()
            
            if not urls:
                return jsonify({"error": "No valid URLs found in file"}), 400
            
            # Crear tarea de análisis
            task_id = str(uuid.uuid4())
            task = AnalysisTask(task_id, urls)
            analysis_tasks[task_id] = task
            
            # Iniciar análisis
            task.start()
            
            return jsonify({
                "task_id": task_id,
                "message": f"Analysis started for {len(urls)} URLs",
                "redirect": url_for('task_status', task_id=task_id)
            })
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze_urls():
    """Maneja el análisis directo de URLs."""
    data = request.get_json()
    
    if not data or 'urls' not in data:
        return jsonify({"error": "No URLs provided"}), 400
    
    urls = data['urls']
    
    if not urls:
        return jsonify({"error": "Empty URL list"}), 400
    
    # Validar URLs
    valid_urls = []
    for url in urls:
        if url.startswith(('http://', 'https://')):
            valid_urls.append(url)
    
    if not valid_urls:
        return jsonify({"error": "No valid URLs found"}), 400
    
    # Crear tarea de análisis
    task_id = str(uuid.uuid4())
    task = AnalysisTask(task_id, valid_urls)
    analysis_tasks[task_id] = task
    
    # Iniciar análisis
    task.start()
    
    return jsonify({
        "task_id": task_id,
        "message": f"Analysis started for {len(valid_urls)} URLs",
        "redirect": url_for('task_status', task_id=task_id)
    })


@app.route('/task/<task_id>')
def task_status(task_id):
    """Página de estado de una tarea."""
    if task_id not in analysis_tasks:
        return render_template('error.html', error="Task not found"), 404
    
    return render_template('task.html', task_id=task_id)


@app.route('/api/task/<task_id>')
def api_task_status(task_id):
    """API para obtener el estado de una tarea."""
    if task_id not in analysis_tasks:
        return jsonify({"error": "Task not found"}), 404
    
    task = analysis_tasks[task_id]
    return jsonify(task.get_status())


@app.route('/api/task/<task_id>/results')
def api_task_results(task_id):
    """API para obtener los resultados de una tarea."""
    if task_id not in analysis_tasks:
        return jsonify({"error": "Task not found"}), 404
    
    task = analysis_tasks[task_id]
    
    if task.status != "completed":
        return jsonify({"error": "Task not completed yet"}), 400
    
    # Simplificar resultados para la API
    simplified_results = {}
    
    for url, result in task.results.items():
        if result.get("success", False):
            simplified_results[url] = {
                "title": result.get("title", ""),
                "success": True,
                "detection_count": len(result.get("detections", [])),
                "pattern_types": list(set(d.get("pattern_type", "") for d in result.get("detections", []))),
                "reports": task.reports.get(url, {})
            }
        else:
            simplified_results[url] = {
                "success": False,
                "error": result.get("error", "Unknown error")
            }
    
    return jsonify(simplified_results)


@app.route('/api/task/<task_id>/report/<path:url>')
def api_url_report(task_id, url):
    """API para obtener el informe de una URL específica."""
    if task_id not in analysis_tasks:
        return jsonify({"error": "Task not found"}), 404
    
    task = analysis_tasks[task_id]
    
    if task.status != "completed":
        return jsonify({"error": "Task not completed yet"}), 400
    
    if url not in task.results:
        return jsonify({"error": "URL not found in task results"}), 404
    
    result = task.results[url]
    
    if not result.get("success", False):
        return jsonify({"error": "Analysis failed for this URL", "details": result.get("error", "")}), 400
    
    # Devolver informe completo
    return jsonify({
        "url": url,
        "title": result.get("title", ""),
        "detections": result.get("detections", []),
        "screenshots": result.get("screenshots", {}),
        "reports": task.reports.get(url, {})
    })


@app.route('/download/<path:filepath>')
def download_file(filepath):
    """Descarga un archivo."""
    # Verificar que el archivo existe y está dentro de los directorios permitidos
    file_path = Path(filepath)
    
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
    
    # Verificar que el archivo está en un directorio permitido
    allowed_dirs = [REPORTS_FOLDER, SCREENSHOTS_FOLDER, EVIDENCE_FOLDER]
    if not any(str(file_path).startswith(str(allowed_dir)) for allowed_dir in allowed_dirs):
        return jsonify({"error": "Access denied"}), 403
    
    return send_file(file_path, as_attachment=True)


@app.route('/dashboard')
def dashboard():
    """Página de dashboard."""
    return render_template('dashboard.html')


@app.route('/api/dashboard/summary')
def api_dashboard_summary():
    """API para obtener un resumen para el dashboard."""
    # Contar tareas completadas
    completed_tasks = [task for task in analysis_tasks.values() if task.status == "completed"]
    
    if not completed_tasks:
        return jsonify({
            "total_tasks": 0,
            "total_urls": 0,
            "total_detections": 0,
            "pattern_distribution": {}
        })
    
    # Calcular estadísticas
    total_urls = sum(task.total_urls for task in completed_tasks)
    total_detections = sum(
        len(result.get("detections", []))
        for task in completed_tasks
        for result in task.results.values()
        if result.get("success", False)
    )
    
    # Calcular distribución de patrones
    pattern_distribution = {
        "confirmshaming": 0,
        "preselection": 0,
        "hidden_costs": 0,
        "difficult_cancellation": 0,
        "misleading_ads": 0,
        "false_urgency": 0,
        "confusing_interface": 0
    }
    
    for task in completed_tasks:
        for result in task.results.values():
            if result.get("success", False):
                for detection in result.get("detections", []):
                    pattern_type = detection.get("pattern_type", "")
                    if pattern_type in pattern_distribution:
                        pattern_distribution[pattern_type] += 1
    
    return jsonify({
        "total_tasks": len(completed_tasks),
        "total_urls": total_urls,
        "total_detections": total_detections,
        "pattern_distribution": pattern_distribution
    })


@app.route('/api/dashboard/recent_detections')
def api_dashboard_recent_detections():
    """API para obtener las detecciones más recientes."""
    # Obtener tareas completadas
    completed_tasks = [task for task in analysis_tasks.values() if task.status == "completed"]
    
    if not completed_tasks:
        return jsonify([])
    
    # Ordenar tareas por fecha de finalización (más recientes primero)
    sorted_tasks = sorted(
        completed_tasks,
        key=lambda t: t.end_time if t.end_time else datetime.min,
        reverse=True
    )
    
    # Recopilar detecciones recientes
    recent_detections = []
    
    for task in sorted_tasks[:5]:  # Limitar a las 5 tareas más recientes
        for url, result in task.results.items():
            if result.get("success", False) and result.get("detections", []):
                for detection in result.get("detections", [])[:3]:  # Limitar a 3 detecciones por URL
                    recent_detections.append({
                        "url": url,
                        "title": result.get("title", "Sin título"),
                        "pattern_type": detection.get("pattern_type", ""),
                        "confidence": detection.get("confidence", 0),
                        "timestamp": task.end_time.isoformat() if task.end_time else None
                    })
    
    # Limitar a las 10 detecciones más recientes
    return jsonify(recent_detections[:10])


@app.route('/api/dashboard/top_sites')
def api_dashboard_top_sites():
    """API para obtener los sitios con más patrones oscuros."""
    # Obtener tareas completadas
    completed_tasks = [task for task in analysis_tasks.values() if task.status == "completed"]
    
    if not completed_tasks:
        return jsonify([])
    
    # Recopilar datos de sitios
    site_data = {}
    
    for task in completed_tasks:
        for url, result in task.results.items():
            if result.get("success", False):
                detections = result.get("detections", [])
                if url not in site_data:
                    site_data[url] = {
                        "url": url,
                        "title": result.get("title", "Sin título"),
                        "detection_count": 0,
                        "pattern_types": set()
                    }
                
                site_data[url]["detection_count"] += len(detections)
                site_data[url]["pattern_types"].update(d.get("pattern_type", "") for d in detections)
    
    # Convertir a lista y ordenar por número de detecciones
    top_sites = list(site_data.values())
    for site in top_sites:
        site["pattern_types"] = list(site["pattern_types"])
    
    top_sites.sort(key=lambda x: x["detection_count"], reverse=True)
    
    # Limitar a los 10 sitios principales
    return jsonify(top_sites[:10])


@app.route('/api/export/<task_id>/<format>')
def api_export_report(task_id, format):
    """API para exportar informes en diferentes formatos."""
    if task_id not in analysis_tasks:
        return jsonify({"error": "Task not found"}), 404
    
    task = analysis_tasks[task_id]
    
    if task.status != "completed":
        return jsonify({"error": "Task not completed yet"}), 400
    
    if format not in ["csv", "json"]:
        return jsonify({"error": "Unsupported format"}), 400
    
    try:
        # Crear gestor de informes
        report_manager = ReportManager(str(REPORTS_FOLDER))
        
        # Preparar datos para el informe
        summary_reports = []
        for url, result in task.results.items():
            if result.get("success", False):
                report_data = {
                    "url": url,
                    "title": result.get("title", "Sin título"),
                    "timestamp": datetime.now().isoformat(),
                    "summary": {
                        "total_patterns_detected": len(result.get("detections", [])),
                        "pattern_types_detected": list(set(d.get("pattern_type", "") for d in result.get("detections", []))),
                        "severity_score": 0  # Se calculará en el report_manager
                    },
                    "patterns": []
                }
                
                # Agrupar detecciones por tipo
                patterns_by_type = {}
                for detection in result.get("detections", []):
                    pattern_type = detection.get("pattern_type", "unknown")
                    if pattern_type not in patterns_by_type:
                        patterns_by_type[pattern_type] = []
                    patterns_by_type[pattern_type].append(detection)
                
                # Añadir información de cada tipo de patrón
                for pattern_type, pattern_detections in patterns_by_type.items():
                    pattern_info = {
                        "type": pattern_type,
                        "count": len(pattern_detections),
                        "detections": pattern_detections
                    }
                    report_data["patterns"].append(pattern_info)
                
                summary_reports.append(report_data)
        
        # Generar informe
        if format == "csv":
            output_file = f"task_{task_id}_summary"
            summary_path = report_manager.generate_summary_report(
                summary_reports, 
                output_file=output_file
            )
            return jsonify({"file": summary_path})
        else:  # json
            output_file = f"task_{task_id}_full"
            json_path = report_manager.save_reports_json(
                summary_reports,
                output_file=output_file
            )
            return jsonify({"file": json_path})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/cleanup')
def api_cleanup_old_tasks():
    """API para limpiar tareas antiguas (solo para administradores)."""
    # Esta función debería estar protegida con autenticación en un entorno real
    
    # Obtener tareas antiguas (más de 24 horas)
    now = datetime.now()
    old_tasks = []
    
    for task_id, task in list(analysis_tasks.items()):
        if task.end_time and (now - task.end_time).total_seconds() > 86400:  # 24 horas
            old_tasks.append(task_id)
            del analysis_tasks[task_id]
    
    return jsonify({
        "success": True,
        "cleaned_tasks": len(old_tasks),
        "remaining_tasks": len(analysis_tasks)
    })


def create_app():
    """Crea y configura la aplicación Flask."""
    return app


if __name__ == '__main__':
    # Crear directorios de plantillas y estáticos si no existen
    templates_dir = Path(__file__).parent / "templates"
    static_dir = Path(__file__).parent / "static"
    
    templates_dir.mkdir(exist_ok=True)
    static_dir.mkdir(exist_ok=True)
    
    # Crear directorios estáticos si no existen
    (static_dir / "css").mkdir(exist_ok=True)
    (static_dir / "js").mkdir(exist_ok=True)
    (static_dir / "img").mkdir(exist_ok=True)
    
    # Iniciar servidor
    app.run(host='0.0.0.0', port=5000, debug=True)
