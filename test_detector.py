"""
Script para probar el sistema de detección de patrones oscuros en sitios web reales.
Utiliza los detectores implementados y genera informes de los resultados.
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Añadir el directorio raíz al path para importar módulos
sys.path.append('.')

from src.utils.url_loader import URLLoader
from src.crawlers.web_crawler import DarkPatternCrawler  # Usamos DarkPatternCrawler, no WebCrawler
from src.detectors.confirmshaming_detector import ConfirmshamingDetector
from src.detectors.preselection_detector import PreselectionDetector
from src.detectors.hidden_costs_detector import HiddenCostsDetector
from src.detectors.difficult_cancellation_detector import DifficultCancellationDetector
from src.detectors.misleading_ads_detector import MisleadingAdsDetector
from src.detectors.false_urgency_detector import FalseUrgencyDetector
from src.detectors.confusing_interface_detector import ConfusingInterfaceDetector
from src.reports.report_generator import ReportGenerator, ReportManager


def analyze_url(url, detectors, report_generator, screenshots_dir, verbose=True):
    """
    Analiza una URL en busca de patrones oscuros.
    
    Args:
        url: URL a analizar
        detectors: Lista de detectores a utilizar
        report_generator: Generador de informes
        screenshots_dir: Directorio para guardar capturas de pantalla
        verbose: Si True, muestra información detallada durante el proceso
        
    Returns:
        dict: Resultados del análisis
    """
    if verbose:
        print(f"Analizando: {url}")
    
    try:
        # Iniciar navegador con DarkPatternCrawler
        with DarkPatternCrawler(headless=True, screenshots_dir=screenshots_dir) as crawler:
            # Navegar a la URL
            if verbose:
                print(f"Navegando a {url}...")
            
            result = crawler.analyze_page(url)
            
            if not result["success"]:
                if verbose:
                    print(f"Error al navegar: {result.get('error', 'Error desconocido')}")
                return result
            
            if verbose:
                print(f"Navegación exitosa")
                print(f"Título de la página: {result.get('title', 'Sin título')}")
            
            # Obtener contenido y estructura DOM
            page_content = crawler.get_page_content()
            dom_structure = result.get('dom_structure', {})
            
            # Ejecutar detectores
            if verbose:
                print("Ejecutando detectores de patrones oscuros...")
            
            all_detections = []
            
            for detector in detectors:
                if verbose:
                    print(f"Ejecutando detector: {detector.name}")
                
                detections = detector.detect(
                    page_content=page_content,
                    dom_structure=dom_structure,
                    screenshot_path=result["screenshots"]["full"],
                    url=url
                )
                
                if detections:
                    if verbose:
                        print(f"  - Se encontraron {len(detections)} instancias de {detector.name}")
                    all_detections.extend(detections)
                else:
                    if verbose:
                        print(f"  - No se encontraron instancias de {detector.name}")
            
            # Añadir detecciones al resultado
            result["detections"] = all_detections
            
            # Generar informe
            if all_detections:
                if verbose:
                    print(f"Se encontraron {len(all_detections)} patrones oscuros en total")
                    print("Generando informe...")
                
                report = report_generator.generate_report(
                    url=url,
                    detections=all_detections,
                    screenshots=result["screenshots"],
                    metadata={"title": result.get("title", "Sin título")}
                )
                
                # Guardar informe en diferentes formatos
                json_path = report_generator.save_report_json(report)
                csv_path = report_generator.save_report_csv(report)
                html_path = report_generator.generate_html_report(report)
                
                if verbose:
                    print(f"Informe JSON guardado en: {json_path}")
                    print(f"Informe CSV guardado en: {csv_path}")
                    print(f"Informe HTML guardado en: {html_path}")
                
                result["reports"] = {
                    "json": json_path,
                    "csv": csv_path,
                    "html": html_path
                }
            else:
                if verbose:
                    print("No se encontraron patrones oscuros")
            
            return result
    
    except Exception as e:
        if verbose:
            print(f"Error durante el análisis: {str(e)}")
        
        return {
            "url": url,
            "success": False,
            "error": str(e)
        }


def main():
    """Función principal para ejecutar pruebas."""
    # Configurar directorios
    base_dir = Path('.')
    data_dir = base_dir / "data"
    reports_dir = data_dir / "reports"
    screenshots_dir = data_dir / "screenshots"
    
    # Asegurar que los directorios existen
    reports_dir.mkdir(exist_ok=True, parents=True)
    screenshots_dir.mkdir(exist_ok=True, parents=True)
    
    # Cargar URLs de prueba
    print("Cargando URLs de prueba...")
    loader = URLLoader(str(data_dir / "test_sites.csv"))
    urls = loader.load()
    print(f"Se cargaron {len(urls)} URLs para pruebas")
    
    # Configurar detectores
    detectors = [
        ConfirmshamingDetector(),
        PreselectionDetector(),
        HiddenCostsDetector(),
        DifficultCancellationDetector(),
        MisleadingAdsDetector(),
        FalseUrgencyDetector(),
        ConfusingInterfaceDetector()
    ]
    
    # Configurar generador de informes
    report_generator = ReportGenerator(str(reports_dir))
    
    # Analizar cada URL
    results = {}
    for url in urls:
        print("\n" + "="*50)
        result = analyze_url(
            url=url,
            detectors=detectors,
            report_generator=report_generator,
            screenshots_dir=str(screenshots_dir)
        )
        results[url] = result
        print("="*50 + "\n")
        
        # Pausa breve entre sitios para no sobrecargar la red
        time.sleep(2)
    
    # Generar informe resumen
    print("\nGenerando informe resumen...")
    
    # Convertir resultados a formato para el informe
    summary_reports = []
    for url, result in results.items():
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
    
    # Crear gestor de informes
    report_manager = ReportManager(str(reports_dir))
    
    # Generar informe resumen
    if summary_reports:
        summary_path = report_manager.generate_summary_report(
            summary_reports, 
            output_file="summary_report"
        )
        print(f"Informe resumen guardado en: {summary_path}")
        
        # Generar datos para dashboard
        dashboard_path = report_manager.generate_dashboard_data(
            summary_reports,
            output_file="dashboard_data"
        )
        print(f"Datos para dashboard guardados en: {dashboard_path}")
    else:
        print("No hay informes exitosos para generar el resumen")
    
    print("\nPruebas completadas")


if __name__ == "__main__":
    main()
