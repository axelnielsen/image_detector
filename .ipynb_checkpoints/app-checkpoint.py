import os
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, jsonify
from werkzeug.utils import secure_filename

from fondo_blanco_detector import FondoBlancoDetector
from persona_detector import PersonaDetector
from cedula_detector import CedulaIdentidadDetector
from rut_detector import RutChilenoDetector
from image_validator import ImageValidator

# Configuración de la aplicación
app = Flask(__name__)
app.secret_key = 'detector_imagenes_secret_key'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB máximo
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Crear directorios si no existen
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Inicializar el validador de imágenes
validator = ImageValidator()

def allowed_file(filename):
    """Verifica si el archivo tiene una extensión permitida."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Página principal."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Maneja la subida de archivos."""
    # Verificar si se envió un archivo
    if 'file' not in request.files:
        flash('No se seleccionó ningún archivo')
        return redirect(request.url)
    
    file = request.files['file']
    
    # Si el usuario no selecciona un archivo, el navegador envía un
    # archivo vacío sin nombre
    if file.filename == '':
        flash('No se seleccionó ningún archivo')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Guardar el archivo
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Procesar la imagen
        try:
            # Validar imagen
            results = validator.validate_image(file_path)
            
            # Generar informe
            report_path = validator.save_validation_report(
                file_path, 
                results, 
                app.config['RESULTS_FOLDER']
            )
            
            # Obtener nombres de archivos para mostrar resultados
            report_filename = os.path.basename(report_path)
            results_dir = os.path.dirname(report_path)
            
            # Buscar la imagen de resultados
            results_image_filename = None
            for f in os.listdir(results_dir):
                if f.endswith('_results.jpg') and f.startswith(os.path.splitext(os.path.basename(report_path))[0].replace('_report', '')):
                    results_image_filename = f
                    break
            
            # Redirigir a la página de resultados
            return redirect(url_for(
                'show_results', 
                filename=unique_filename,
                report=report_filename,
                results_image=results_image_filename
            ))
            
        except Exception as e:
            flash(f'Error al procesar la imagen: {str(e)}')
            return redirect(url_for('index'))
    
    flash('Tipo de archivo no permitido')
    return redirect(url_for('index'))

@app.route('/results')
def show_results():
    """Muestra los resultados del análisis."""
    filename = request.args.get('filename', '')
    report = request.args.get('report', '')
    results_image = request.args.get('results_image', '')
    
    if not filename or not report:
        flash('Información de resultados incompleta')
        return redirect(url_for('index'))
    
    # Obtener la ruta del informe HTML
    report_path = os.path.join(app.config['RESULTS_FOLDER'], report)
    
    # Si el informe existe, mostrar su contenido
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        return render_template(
            'results.html', 
            filename=filename,
            report=report,
            results_image=results_image,
            report_content=report_content
        )
    
    flash('No se encontró el informe de resultados')
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Sirve los archivos subidos."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    """Sirve los archivos de resultados."""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/about')
def about():
    """Página de información sobre el sistema."""
    return render_template('about.html')

@app.route('/api/validate', methods=['POST'])
def api_validate():
    """API para validar imágenes programáticamente."""
    # Verificar si se envió un archivo
    if 'file' not in request.files:
        return jsonify({'error': 'No se envió ningún archivo'}), 400
    
    file = request.files['file']
    
    # Si el usuario no selecciona un archivo
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    
    if file and allowed_file(file.filename):
        # Guardar el archivo
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Procesar la imagen
        try:
            # Validar imagen
            results = validator.validate_image(file_path)
            
            # Convertir valores no serializables a tipos nativos de Python
            serializable_results = {}
            for key, value in results.items():
                if key == 'confidence':
                    serializable_results[key] = {k: float(v) for k, v in value.items()}
                elif key == 'metadata':
                    serializable_results[key] = value
                else:
                    serializable_results[key] = bool(value) if isinstance(value, (bool, np.bool_)) else value
            
            return jsonify({
                'success': True,
                'filename': unique_filename,
                'results': serializable_results
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Tipo de archivo no permitido'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
