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
