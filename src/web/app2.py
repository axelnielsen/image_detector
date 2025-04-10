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
