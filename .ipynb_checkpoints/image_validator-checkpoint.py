import os
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

from fondo_blanco_detector import FondoBlancoDetector
from persona_detector import PersonaDetector
from cedula_detector import CedulaIdentidadDetector
from rut_detector import RutChilenoDetector
from model_trainer import ModelTrainer

class ImageValidator:
    """
    Sistema para validar características en imágenes utilizando los detectores
    implementados y/o el modelo entrenado.
    """
    
    def __init__(self, model_dir=None):
        """
        Inicializa el validador de imágenes.
        
        Args:
            model_dir (str, optional): Directorio donde se encuentran los modelos entrenados.
                                      Si es None, se utilizan solo los detectores individuales.
        """
        # Inicializar detectores individuales
        self.fondo_blanco_detector = FondoBlancoDetector()
        self.persona_detector = PersonaDetector()
        self.cedula_detector = CedulaIdentidadDetector()
        self.rut_detector = RutChilenoDetector()
        
        # Inicializar trainer (para usar modelo entrenado si está disponible)
        if model_dir is not None:
            self.model_dir = Path(model_dir)
            self.trainer = ModelTrainer(
                dataset_dir="/home/ubuntu/detector_imagenes/dataset",
                model_dir=model_dir
            )
        else:
            self.trainer = None
    
    def validate_image(self, image_path, use_model=False):
        """
        Valida una imagen utilizando los detectores individuales o el modelo entrenado.
        
        Args:
            image_path (str): Ruta de la imagen a validar.
            use_model (bool): Si es True, intenta usar el modelo entrenado. Si es False o
                             el modelo no está disponible, usa los detectores individuales.
            
        Returns:
            dict: Resultados de la validación.
        """
        # Cargar imagen
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # Si se solicita usar el modelo entrenado y está disponible
        if use_model and self.trainer is not None:
            try:
                # Intentar usar el modelo entrenado
                results = self.trainer.predict(image_path)
                return results
            except Exception as e:
                print(f"Error al usar el modelo entrenado: {e}")
                print("Usando detectores individuales como respaldo...")
        
        # Usar detectores individuales
        fondo_result = self.fondo_blanco_detector.detect(image)
        persona_result = self.persona_detector.detect(image)
        cedula_result = self.cedula_detector.detect(image)
        rut_result = self.rut_detector.detect(image)
        
        # Formatear resultados
        results = {
            'fondo_blanco': fondo_result["present"],
            'persona': persona_result["present"],
            'cedula': cedula_result["present"],
            'rut': rut_result["present"],
            'confidence': {
                'fondo_blanco': fondo_result["confidence"],
                'persona': persona_result["confidence"],
                'cedula': cedula_result["confidence"],
                'rut': rut_result["confidence"]
            },
            'metadata': {
                'fondo_blanco': fondo_result.get("metadata", {}),
                'rut': rut_result.get("rut", None)
            }
        }
        
        return results
    
    def visualize_results(self, image_path, results):
        """
        Genera una visualización de los resultados de la validación.
        
        Args:
            image_path (str): Ruta de la imagen validada.
            results (dict): Resultados de la validación.
            
        Returns:
            np.array: Imagen con visualización de resultados.
        """
        # Cargar imagen
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # Crear una copia de la imagen
        vis_image = image.copy()
        
        # Definir colores para cada característica
        colors = {
            'fondo_blanco': (255, 0, 0),    # Azul
            'persona': (0, 255, 0),         # Verde
            'cedula': (0, 0, 255),          # Rojo
            'rut': (255, 255, 0)            # Cian
        }
        
        # Añadir título
        cv2.putText(
            vis_image, 
            "Resultados de Validación", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.0, 
            (255, 255, 255), 
            2
        )
        
        # Añadir resultados para cada característica
        y_pos = 70
        for feature, present in results.items():
            if feature not in ['confidence', 'metadata']:
                # Obtener estado y confianza
                is_present = results[feature]
                confidence = results['confidence'][feature]
                
                # Definir color (verde si es positivo, rojo si es negativo)
                color = (0, 255, 0) if is_present else (0, 0, 255)
                
                # Añadir texto
                feature_name = feature.replace('_', ' ').title()
                text = f"{feature_name}: {'Sí' if is_present else 'No'} ({confidence:.2f})"
                cv2.putText(vis_image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                y_pos += 40
        
        # Añadir metadatos adicionales si están disponibles
        if 'metadata' in results:
            if 'fondo_blanco' in results['metadata'] and 'porcentaje_blanco' in results['metadata']['fondo_blanco']:
                text = f"% Blanco: {results['metadata']['fondo_blanco']['porcentaje_blanco']:.1f}%"
                cv2.putText(vis_image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_pos += 40
            
            if 'rut' in results['metadata'] and results['metadata']['rut'] is not None:
                text = f"RUT: {results['metadata']['rut']}"
                cv2.putText(vis_image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_image
    
    def save_validation_report(self, image_path, results, output_dir):
        """
        Guarda un informe de validación con los resultados y la visualización.
        
        Args:
            image_path (str): Ruta de la imagen validada.
            results (dict): Resultados de la validación.
            output_dir (str): Directorio donde guardar el informe.
            
        Returns:
            str: Ruta del informe generado.
        """
        # Crear directorio de salida si no existe
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generar nombre de archivo basado en timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(image_path).stem
        report_name = f"{base_name}_{timestamp}"
        
        # Generar visualización
        vis_image = self.visualize_results(image_path, results)
        
        # Guardar imagen con resultados
        vis_path = output_dir / f"{report_name}_results.jpg"
        cv2.imwrite(str(vis_path), vis_image)
        
        # Guardar resultados en JSON
        json_path = output_dir / f"{report_name}_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            # Convertir valores no serializables a tipos nativos de Python
            serializable_results = {}
            for key, value in results.items():
                if key == 'confidence':
                    serializable_results[key] = {k: float(v) for k, v in value.items()}
                elif key == 'metadata':
                    serializable_results[key] = value
                else:
                    serializable_results[key] = bool(value) if isinstance(value, (bool, np.bool_)) else value
            
            json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)
        
        # Crear informe HTML
        html_path = output_dir / f"{report_name}_report.html"
        self._generate_html_report(image_path, results, vis_path, html_path)
        
        return str(html_path)
    
    def _generate_html_report(self, image_path, results, vis_path, html_path):
        """
        Genera un informe HTML con los resultados de la validación.
        
        Args:
            image_path (str): Ruta de la imagen validada.
            results (dict): Resultados de la validación.
            vis_path (str): Ruta de la imagen con visualización.
            html_path (str): Ruta donde guardar el informe HTML.
        """
        # Obtener nombres de archivo
        image_filename = Path(image_path).name
        vis_filename = Path(vis_path).name
        
        # Crear contenido HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Informe de Validación de Imagen</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .container {{ max-width: 1000px; margin: 0 auto; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .results {{ display: flex; flex-wrap: wrap; }}
                .result-card {{ background-color: #f9f9f9; border-radius: 5px; padding: 15px; margin: 10px; flex: 1; min-width: 200px; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .images {{ display: flex; flex-wrap: wrap; justify-content: space-between; margin-top: 20px; }}
                .image-container {{ flex: 1; min-width: 300px; margin: 10px; }}
                img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Informe de Validación de Imagen</h1>
                    <p>Fecha: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</p>
                    <p>Imagen: {image_filename}</p>
                </div>
                
                <h2>Resultados de la Validación</h2>
                <div class="results">
        """
        
        # Añadir tarjetas de resultados
        for feature, present in results.items():
            if feature not in ['confidence', 'metadata']:
                is_present = results[feature]
                confidence = results['confidence'][feature]
                feature_name = feature.replace('_', ' ').title()
                
                html_content += f"""
                    <div class="result-card">
                        <h3>{feature_name}</h3>
                        <p class="{'positive' if is_present else 'negative'}">
                            <strong>{'Detectado' if is_present else 'No Detectado'}</strong>
                        </p>
                        <p>Confianza: {confidence:.2f}</p>
                    </div>
                """
        
        html_content += """
                </div>
                
                <h2>Detalles</h2>
                <table>
                    <tr>
                        <th>Característica</th>
                        <th>Resultado</th>
                        <th>Confianza</th>
                        <th>Detalles</th>
                    </tr>
        """
        
        # Añadir filas de la tabla
        for feature, present in results.items():
            if feature not in ['confidence', 'metadata']:
                is_present = results[feature]
                confidence = results['confidence'][feature]
                feature_name = feature.replace('_', ' ').title()
                
                # Obtener detalles adicionales
                details = ""
                if feature == 'fondo_blanco' and 'metadata' in results and 'fondo_blanco' in results['metadata']:
                    if 'porcentaje_blanco' in results['metadata']['fondo_blanco']:
                        details = f"Porcentaje de blanco: {results['metadata']['fondo_blanco']['porcentaje_blanco']:.1f}%"
                
                if feature == 'rut' and 'metadata' in results and 'rut' in results['metadata'] and results['metadata']['rut']:
                    details = f"RUT detectado: {results['metadata']['rut']}"
                
                html_content += f"""
                    <tr>
                        <td>{feature_name}</td>
                        <td class="{'positive' if is_present else 'negative'}">{'Sí' if is_present else 'No'}</td>
                        <td>{confidence:.2f}</td>
                        <td>{details}</td>
                    </tr>
                """
        
        html_content += f"""
                </table>
                
                <h2>Imágenes</h2>
                <div class="images">
                    <div class="image-container">
                        <h3>Imagen Original</h3>
                        <img src="{image_filename}" alt="Imagen Original">
                    </div>
                    <div class="image-container">
                        <h3>Resultados Visualizados</h3>
                        <img src="{vis_filename}" alt="Resultados Visualizados">
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Guardar archivo HTML
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

def main():
    """
    Función principal para ejecutar el validador de imágenes desde línea de comandos.
    """
    parser = argparse.ArgumentParser(description='Validador de Características en Imágenes')
    parser.add_argument('image_path', type=str, help='Ruta de la imagen a validar')
    parser.add_argument('--model_dir', type=str, default=None, help='Directorio de modelos entrenados')
    parser.add_argument('--use_model', action='store_true', help='Usar modelo entrenado si está disponible')
    parser.add_argument('--output_dir', type=str, default='./resultados', help='Directorio para guardar resultados')
    
    args = parser.parse_args()
    
    # Crear validador
    validator = ImageValidator(model_dir=args.model_dir)
    
    # Validar imagen
    results = validator.validate_image(args.image_path, use_model=args.use_model)
    
    # Guardar informe
    report_path = validator.save_validation_report(args.image_path, results, args.output_dir)
    
    print(f"Resultados guardados en: {report_path}")
    print("\nResumen de resultados:")
    for feature, present in results.items():
        if feature not in ['confidence', 'metadata']:
            print(f"- {feature}: {'Sí' if present else 'No'} (Confianza: {results['confidence'][feature]:.2f})")

if __name__ == "__main__":
    main()
