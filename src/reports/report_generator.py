"""
Módulo para generar informes sobre patrones oscuros detectados.
"""

import os
import json
import csv
import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class ReportGenerator:
    """Clase para generar informes sobre patrones oscuros detectados."""
    
    def __init__(self, output_dir: str = None):
        """
        Inicializa el generador de informes.
        
        Args:
            output_dir: Directorio donde se guardarán los informes
        """
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.join(os.getcwd(), 'data', 'reports')
        
        # Crear directorio si no existe
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_report(self, url: str, detections: List[Dict[str, Any]], 
                       screenshots: Dict[str, str], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera un informe completo para una URL.
        
        Args:
            url: URL analizada
            detections: Lista de patrones oscuros detectados
            screenshots: Diccionario de capturas de pantalla
            metadata: Metadatos adicionales (título, fecha, etc.)
            
        Returns:
            Dict[str, Any]: Informe completo
        """
        # Agrupar detecciones por tipo de patrón
        patterns_by_type = {}
        for detection in detections:
            pattern_type = detection.get("pattern_type", "unknown")
            if pattern_type not in patterns_by_type:
                patterns_by_type[pattern_type] = []
            patterns_by_type[pattern_type].append(detection)
        
        # Crear estructura del informe
        report = {
            "url": url,
            "timestamp": datetime.datetime.now().isoformat(),
            "title": metadata.get("title", ""),
            "summary": {
                "total_patterns_detected": len(detections),
                "pattern_types_detected": list(patterns_by_type.keys()),
                "severity_score": self._calculate_severity_score(detections)
            },
            "screenshots": screenshots,
            "patterns": []
        }
        
        # Añadir detalles de cada tipo de patrón
        for pattern_type, pattern_detections in patterns_by_type.items():
            pattern_info = {
                "type": pattern_type,
                "count": len(pattern_detections),
                "detections": pattern_detections,
                "improvement_suggestions": self._generate_improvement_suggestions(pattern_type, pattern_detections)
            }
            report["patterns"].append(pattern_info)
        
        return report
    
    def save_report_json(self, report: Dict[str, Any], filename: str = None) -> str:
        """
        Guarda el informe en formato JSON.
        
        Args:
            report: Informe a guardar
            filename: Nombre del archivo (sin extensión)
            
        Returns:
            str: Ruta al archivo guardado
        """
        if not filename:
            # Generar nombre basado en la URL y timestamp
            url_part = report["url"].split("//")[1].split("/")[0].replace(".", "_")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{url_part}_{timestamp}"
        
        # Asegurar que el nombre no contiene caracteres inválidos
        filename = ''.join(c if c.isalnum() or c in '_-' else '_' for c in filename)
        
        # Crear ruta completa
        file_path = os.path.join(self.output_dir, f"{filename}.json")
        
        # Guardar informe
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return file_path
    
    def save_report_csv(self, report: Dict[str, Any], filename: str = None) -> str:
        """
        Guarda un resumen del informe en formato CSV.
        
        Args:
            report: Informe a guardar
            filename: Nombre del archivo (sin extensión)
            
        Returns:
            str: Ruta al archivo guardado
        """
        if not filename:
            # Generar nombre basado en la URL y timestamp
            url_part = report["url"].split("//")[1].split("/")[0].replace(".", "_")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{url_part}_{timestamp}"
        
        # Asegurar que el nombre no contiene caracteres inválidos
        filename = ''.join(c if c.isalnum() or c in '_-' else '_' for c in filename)
        
        # Crear ruta completa
        file_path = os.path.join(self.output_dir, f"{filename}.csv")
        
        # Preparar datos para CSV
        rows = []
        
        # Fila de encabezado
        header = ["URL", "Título", "Fecha", "Total Patrones", "Puntuación de Severidad", 
                 "Tipo de Patrón", "Número de Detecciones", "Ubicación", "Confianza", "Sugerencia de Mejora"]
        rows.append(header)
        
        # Filas de datos
        for pattern in report["patterns"]:
            pattern_type = pattern["type"]
            for detection in pattern["detections"]:
                row = [
                    report["url"],
                    report["title"],
                    report["timestamp"],
                    report["summary"]["total_patterns_detected"],
                    report["summary"]["severity_score"],
                    pattern_type,
                    1,  # Cada fila es una detección
                    detection.get("location", ""),
                    detection.get("confidence", 0),
                    pattern["improvement_suggestions"][0] if pattern["improvement_suggestions"] else ""
                ]
                rows.append(row)
        
        # Guardar CSV
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        return file_path
    
    def generate_html_report(self, report: Dict[str, Any], filename: str = None) -> str:
        """
        Genera un informe en formato HTML.
        
        Args:
            report: Informe a convertir a HTML
            filename: Nombre del archivo (sin extensión)
            
        Returns:
            str: Ruta al archivo HTML generado
        """
        if not filename:
            # Generar nombre basado en la URL y timestamp
            url_part = report["url"].split("//")[1].split("/")[0].replace(".", "_")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{url_part}_{timestamp}"
        
        # Asegurar que el nombre no contiene caracteres inválidos
        filename = ''.join(c if c.isalnum() or c in '_-' else '_' for c in filename)
        
        # Crear ruta completa
        file_path = os.path.join(self.output_dir, f"{filename}.html")
        
        # Generar HTML
        html = self._generate_html_content(report)
        
        # Guardar HTML
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return file_path
    
    def _calculate_severity_score(self, detections: List[Dict[str, Any]]) -> float:
        """
        Calcula una puntuación de severidad basada en las detecciones.
        
        Args:
            detections: Lista de patrones oscuros detectados
            
        Returns:
            float: Puntuación de severidad (0-10)
        """
        if not detections:
            return 0.0
        
        # Pesos por tipo de patrón
        pattern_weights = {
            "confirmshaming": 0.8,
            "preselection": 0.7,
            "hidden_costs": 0.9,
            "difficult_cancellation": 0.9,
            "misleading_ads": 0.8,
            "false_urgency": 0.7,
            "confusing_interface": 0.8,
            "unknown": 0.5
        }
        
        # Calcular puntuación
        total_weight = 0
        weighted_sum = 0
        
        for detection in detections:
            pattern_type = detection.get("pattern_type", "unknown")
            confidence = detection.get("confidence", 0.5)
            
            weight = pattern_weights.get(pattern_type, 0.5)
            total_weight += weight
            weighted_sum += weight * confidence
        
        # Normalizar a escala 0-10
        if total_weight > 0:
            score = (weighted_sum / total_weight) * 10
        else:
            score = 0
        
        return round(score, 1)
    
    def _generate_improvement_suggestions(self, pattern_type: str, 
                                         detections: List[Dict[str, Any]]) -> List[str]:
        """
        Genera sugerencias de mejora para un tipo de patrón.
        
        Args:
            pattern_type: Tipo de patrón oscuro
            detections: Lista de detecciones de ese tipo
            
        Returns:
            List[str]: Lista de sugerencias de mejora
        """
        # Sugerencias genéricas por tipo de patrón
        generic_suggestions = {
            "confirmshaming": [
                "Utilice un lenguaje neutral para las opciones de rechazo.",
                "Evite hacer que los usuarios se sientan culpables por declinar.",
                "Presente todas las opciones con igual respeto hacia la decisión del usuario."
            ],
            "preselection": [
                "Las opciones que implican costos adicionales o compartir datos no deberían estar preseleccionadas.",
                "Permita que los usuarios elijan activamente todas las opciones.",
                "Mantenga todas las casillas desmarcadas por defecto, especialmente para suscripciones y marketing."
            ],
            "hidden_costs": [
                "Muestre todos los costos desde el principio del proceso.",
                "Evite añadir cargos sorpresa en las últimas etapas.",
                "Sea transparente sobre todos los costos adicionales como envío, impuestos o comisiones."
            ],
            "difficult_cancellation": [
                "Haga que el proceso de cancelación sea tan sencillo como el de suscripción.",
                "Proporcione un enlace directo a la cancelación en áreas visibles.",
                "Evite procesos de múltiples pasos o llamadas telefónicas obligatorias para cancelar."
            ],
            "misleading_ads": [
                "Distinga claramente entre contenido publicitario y contenido orgánico.",
                "Etiquete todos los anuncios de manera visible y clara.",
                "Evite diseños que confundan anuncios con funcionalidades del sitio."
            ],
            "false_urgency": [
                "Utilice indicadores de urgencia solo cuando sean reales.",
                "Evite contadores falsos o mensajes de escasez fabricados.",
                "Sea honesto sobre la disponibilidad real de productos o servicios."
            ],
            "confusing_interface": [
                "Diseñe interfaces claras con jerarquía visual adecuada.",
                "Asegúrese de que los botones de acción principal y secundaria sean visualmente distintos.",
                "Utilice etiquetas claras y descriptivas para todos los elementos interactivos."
            ]
        }
        
        # Obtener sugerencias específicas de las detecciones
        specific_suggestions = []
        for detection in detections:
            if "improvement_suggestion" in detection:
                specific_suggestions.append(detection["improvement_suggestion"])
        
        # Combinar sugerencias genéricas y específicas, eliminando duplicados
        all_suggestions = specific_suggestions + generic_suggestions.get(pattern_type, [])
        unique_suggestions = list(dict.fromkeys(all_suggestions))
        
        return unique_suggestions
    
    def _generate_html_content(self, report: Dict[str, Any]) -> str:
        """
        Genera el contenido HTML para el informe.
        
        Args:
            report: Informe a convertir a HTML
            
        Returns:
            str: Contenido HTML
        """
        # Función para escapar caracteres HTML
        def escape_html(text):
            return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")
        
        # Convertir timestamp a formato legible
        try:
            timestamp = datetime.datetime.fromisoformat(report["timestamp"]).strftime("%d/%m/%Y %H:%M:%S")
        except:
            timestamp = report["timestamp"]
        
        # Generar HTML
        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Informe de Patrones Oscuros - {escape_html(report["url"])}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .header {{
            border-bottom: 2px solid #eee;
            padding-bottom: 20px;
            margin-bottom: 20px;
        }}
        .summary {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .severity {{
            font-size: 24px;
            font-weight: bold;
        }}
        .severity-high {{
            color: #e74c3c;
        }}
        .severity-medium {{
            color: #f39c12;
        }}
        .severity-low {{
            color: #27ae60;
        }}
        .pattern-section {{
            margin-bottom: 30px;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
        }}
        .pattern-header {{
            background-color: #eee;
            padding: 10px;
            margin: -15px -15px 15px -15px;
            border-radius: 5px 5px 0 0;
        }}
        .detection {{
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px dashed #ddd;
        }}
        .detection:last-child {{
            border-bottom: none;
        }}
        .suggestions {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
        }}
        .screenshot {{
            max-width: 100%;
            height: auto;
            margin: 10px 0;
            border: 1px solid #ddd;
        }}
        .evidence {{
            background-color: #f8f8f8;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            white-space: pre-wrap;
            margin: 10px 0;
        }}
        .confidence {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 14px;
            font-weight: bold;
        }}
        .confidence-high {{
            background-color: #e74c3c;
            color: white;
        }}
        .confidence-medium {{
            background-color: #f39c12;
            color: white;
        }}
        .confidence-low {{
            background-color: #27ae60;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Informe de Detección de Patrones Oscuros</h1>
        <p><strong>URL:</strong> <a href="{escape_html(report["url"])}" target="_blank">{escape_html(report["url"])}</a></p>
        <p><strong>Título:</strong> {escape_html(report["title"])}</p>
        <p><strong>Fecha de análisis:</strong> {escape_html(timestamp)}</p>
    </div>
    
    <div class="summary">
        <h2>Resumen</h2>
        <p>Se detectaron <strong>{report["summary"]["total_patterns_detected"]}</strong> instancias de patrones oscuros.</p>
        <p>Tipos de patrones detectados: <strong>{", ".join(report["summary"]["pattern_types_detected"])}</strong></p>
        
        <p>Puntuación de severidad: 
            <span class="severity {
                'severity-high' if report["summary"]["severity_score"] >= 7 else
                'severity-medium' if report["summary"]["severity_score"] >= 4 else
                'severity-low'
            }">{report["summary"]["severity_score"]}/10</span>
        </p>
    </div>
    
    <h2>Patrones Detectados</h2>
"""
        
        # Añadir secciones para cada tipo de patrón
        for pattern in report["patterns"]:
            pattern_type = pattern["type"]
            pattern_count = pattern["count"]
            
            # Mapear tipos de patrones a nombres más amigables
            pattern_names = {
                "confirmshaming": "Confirmshaming (Avergonzar al usuario)",
                "preselection": "Preselección de opciones",
                "hidden_costs": "Cargos ocultos",
                "difficult_cancellation": "Suscripciones difíciles de cancelar",
                "misleading_ads": "Publicidad engañosa",
                "false_urgency": "Falsos contadores de urgencia o escasez",
                "confusing_interface": "Interfaces confusas o botones engañosos"
            }
            
            pattern_name = pattern_names.get(pattern_type, pattern_type.replace("_", " ").title())
            
            html += f"""
    <div class="pattern-section">
        <div class="pattern-header">
            <h3>{escape_html(pattern_name)} ({pattern_count} {'detección' if pattern_count == 1 else 'detecciones'})</h3>
        </div>
        
        <div class="detections">
"""
            
            # Añadir cada detección
            for detection in pattern["detections"]:
                confidence = detection.get("confidence", 0)
                confidence_class = (
                    "confidence-high" if confidence >= 0.8 else
                    "confidence-medium" if confidence >= 0.6 else
                    "confidence-low"
                )
                
                html += f"""
            <div class="detection">
                <p><strong>Ubicación:</strong> {escape_html(detection.get("location", "No especificada"))}</p>
                <p><strong>Confianza:</strong> <span class="confidence {confidence_class}">{int(confidence * 100)}%</span></p>
"""
                
                # Añadir evidencia
                if "evidence" in detection:
                    evidence = detection["evidence"]
                    if isinstance(evidence, dict):
                        # Formatear la evidencia como JSON
                        evidence_str = json.dumps(evidence, indent=2, ensure_ascii=False)
                        html += f"""
                <p><strong>Evidencia:</strong></p>
                <div class="evidence">{escape_html(evidence_str)}</div>
"""
                    else:
                        html += f"""
                <p><strong>Evidencia:</strong> {escape_html(str(evidence))}</p>
"""
                
                # Añadir captura de pantalla si está disponible
                if "screenshot" in detection and detection["screenshot"]:
                    screenshot = detection["screenshot"]
                    # Convertir ruta absoluta a relativa si es necesario
                    screenshot_filename = os.path.basename(screenshot)
                    html += f"""
                <p><strong>Captura de pantalla:</strong></p>
                <img src="{escape_html(screenshot)}" alt="Evidencia visual" class="screenshot">
"""
                
                html += """
            </div>
"""
            
            # Añadir sugerencias de mejora
            html += """
        </div>
        
        <div class="suggestions">
            <h4>Sugerencias de mejora:</h4>
            <ul>
"""
            
            for suggestion in pattern["improvement_suggestions"]:
                html += f"""
                <li>{escape_html(suggestion)}</li>
"""
            
            html += """
            </ul>
        </div>
    </div>
"""
        
        # Cerrar HTML
        html += """
    <div class="footer">
        <p>Este informe fue generado automáticamente por el Sistema de Detección de Patrones Oscuros.</p>
    </div>
</body>
</html>
"""
        
        return html


class ReportManager:
    """Clase para gestionar múltiples informes y generar resúmenes."""
    
    def __init__(self, reports_dir: str = None):
        """
        Inicializa el gestor de informes.
        
        Args:
            reports_dir: Directorio donde se encuentran los informes
        """
        if reports_dir:
            self.reports_dir = reports_dir
        else:
            self.reports_dir = os.path.join(os.getcwd(), 'data', 'reports')
        
        self.report_generator = ReportGenerator(self.reports_dir)
    
    def load_report(self, file_path: str) -> Dict[str, Any]:
        """
        Carga un informe desde un archivo JSON.
        
        Args:
            file_path: Ruta al archivo JSON
            
        Returns:
            Dict[str, Any]: Informe cargado
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def generate_summary_report(self, reports: List[Dict[str, Any]], 
                               output_file: str = "summary_report") -> str:
        """
        Genera un informe resumen de múltiples informes.
        
        Args:
            reports: Lista de informes
            output_file: Nombre del archivo de salida (sin extensión)
            
        Returns:
            str: Ruta al archivo CSV generado
        """
        # Crear ruta completa
        file_path = os.path.join(self.reports_dir, f"{output_file}.csv")
        
        # Preparar datos para CSV
        rows = []
        
        # Fila de encabezado
        header = ["URL", "Título", "Fecha", "Total Patrones", "Puntuación de Severidad", 
                 "Confirmshaming", "Preselección", "Cargos Ocultos", "Difícil Cancelación", 
                 "Publicidad Engañosa", "Falsa Urgencia", "Interfaz Confusa"]
        rows.append(header)
        
        # Filas de datos
        for report in reports:
            # Contar detecciones por tipo
            pattern_counts = {
                "confirmshaming": 0,
                "preselection": 0,
                "hidden_costs": 0,
                "difficult_cancellation": 0,
                "misleading_ads": 0,
                "false_urgency": 0,
                "confusing_interface": 0
            }
            
            for pattern in report.get("patterns", []):
                pattern_type = pattern.get("type", "")
                if pattern_type in pattern_counts:
                    pattern_counts[pattern_type] = pattern.get("count", 0)
            
            # Crear fila
            row = [
                report.get("url", ""),
                report.get("title", ""),
                report.get("timestamp", ""),
                report.get("summary", {}).get("total_patterns_detected", 0),
                report.get("summary", {}).get("severity_score", 0),
                pattern_counts["confirmshaming"],
                pattern_counts["preselection"],
                pattern_counts["hidden_costs"],
                pattern_counts["difficult_cancellation"],
                pattern_counts["misleading_ads"],
                pattern_counts["false_urgency"],
                pattern_counts["confusing_interface"]
            ]
            rows.append(row)
        
        # Guardar CSV
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        return file_path
    
    def generate_dashboard_data(self, reports: List[Dict[str, Any]], 
                               output_file: str = "dashboard_data") -> str:
        """
        Genera datos para un dashboard web.
        
        Args:
            reports: Lista de informes
            output_file: Nombre del archivo de salida (sin extensión)
            
        Returns:
            str: Ruta al archivo JSON generado
        """
        # Preparar datos para el dashboard
        dashboard_data = {
            "summary": {
                "total_sites": len(reports),
                "total_patterns": sum(r.get("summary", {}).get("total_patterns_detected", 0) for r in reports),
                "average_severity": sum(r.get("summary", {}).get("severity_score", 0) for r in reports) / len(reports) if reports else 0
            },
            "pattern_distribution": {
                "confirmshaming": 0,
                "preselection": 0,
                "hidden_costs": 0,
                "difficult_cancellation": 0,
                "misleading_ads": 0,
                "false_urgency": 0,
                "confusing_interface": 0
            },
            "sites": []
        }
        
        # Calcular distribución de patrones y datos por sitio
        for report in reports:
            site_data = {
                "url": report.get("url", ""),
                "title": report.get("title", ""),
                "severity": report.get("summary", {}).get("severity_score", 0),
                "total_patterns": report.get("summary", {}).get("total_patterns_detected", 0),
                "patterns": {}
            }
            
            for pattern in report.get("patterns", []):
                pattern_type = pattern.get("type", "")
                count = pattern.get("count", 0)
                
                if pattern_type in dashboard_data["pattern_distribution"]:
                    dashboard_data["pattern_distribution"][pattern_type] += count
                    site_data["patterns"][pattern_type] = count
            
            dashboard_data["sites"].append(site_data)
        
        # Crear ruta completa
        file_path = os.path.join(self.reports_dir, f"{output_file}.json")
        
        # Guardar JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
        
        return file_path
