import os
import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

from fondo_blanco_detector import FondoBlancoDetector
from persona_detector import PersonaDetector
from cedula_detector import CedulaIdentidadDetector
from rut_detector import RutChilenoDetector
from image_validator import ImageValidator

def test_detectors():
    """
    Prueba los detectores individuales con las imágenes de prueba.
    """
    # Directorios
    test_dir = Path("/home/ubuntu/detector_imagenes/test_images")
    results_dir = Path("/home/ubuntu/detector_imagenes/test_results")
    results_dir.mkdir(exist_ok=True)
    
    # Inicializar detectores
    fondo_detector = FondoBlancoDetector()
    persona_detector = PersonaDetector()
    cedula_detector = CedulaIdentidadDetector()
    rut_detector = RutChilenoDetector()
    
    # Resultados
    results = {
        "fondo_blanco": [],
        "persona": [],
        "cedula": [],
        "rut": []
    }
    
    # Probar detector de fondo blanco
    print("\n=== Prueba de Detector de Fondo Blanco ===")
    for img_path in test_dir.glob("*.jpg"):
        img = cv2.imread(str(img_path))
        result = fondo_detector.detect(img)
        
        print(f"Imagen: {img_path.name}")
        print(f"  Fondo blanco: {'Sí' if result['present'] else 'No'}")
        print(f"  Confianza: {result['confidence']:.2f}")
        print(f"  Porcentaje blanco: {result['metadata']['porcentaje_blanco']:.1f}%")
        
        # Guardar resultado
        results["fondo_blanco"].append({
            "image": img_path.name,
            "result": result
        })
        
        # Visualizar resultado
        vis_img = fondo_detector.visualize(img, result)
        output_path = results_dir / f"fondo_{img_path.name}"
        cv2.imwrite(str(output_path), vis_img)
    
    # Probar detector de personas
    print("\n=== Prueba de Detector de Personas ===")
    for img_path in test_dir.glob("*.jpg"):
        img = cv2.imread(str(img_path))
        result = persona_detector.detect(img)
        
        print(f"Imagen: {img_path.name}")
        print(f"  Persona: {'Sí' if result['present'] else 'No'}")
        print(f"  Confianza: {result['confidence']:.2f}")
        
        # Guardar resultado
        results["persona"].append({
            "image": img_path.name,
            "result": result
        })
        
        # Visualizar resultado
        vis_img = persona_detector.visualize(img, result)
        output_path = results_dir / f"persona_{img_path.name}"
        cv2.imwrite(str(output_path), vis_img)
    
    # Probar detector de cédulas
    print("\n=== Prueba de Detector de Cédulas ===")
    for img_path in test_dir.glob("*.jpg"):
        img = cv2.imread(str(img_path))
        result = cedula_detector.detect(img)
        
        print(f"Imagen: {img_path.name}")
        print(f"  Cédula: {'Sí' if result['present'] else 'No'}")
        print(f"  Confianza: {result['confidence']:.2f}")
        
        # Guardar resultado
        results["cedula"].append({
            "image": img_path.name,
            "result": result
        })
        
        # Visualizar resultado
        vis_img = cedula_detector.visualize(img, result)
        output_path = results_dir / f"cedula_{img_path.name}"
        cv2.imwrite(str(output_path), vis_img)
    
    # Probar detector de RUT
    print("\n=== Prueba de Detector de RUT ===")
    for img_path in test_dir.glob("*.jpg"):
        img = cv2.imread(str(img_path))
        result = rut_detector.detect(img)
        
        print(f"Imagen: {img_path.name}")
        print(f"  RUT: {'Sí' if result['present'] else 'No'}")
        print(f"  Confianza: {result['confidence']:.2f}")
        if result["present"] and "rut" in result:
            print(f"  RUT detectado: {result['rut']}")
        
        # Guardar resultado
        results["rut"].append({
            "image": img_path.name,
            "result": result
        })
        
        # Visualizar resultado
        vis_img = rut_detector.visualize(img, result)
        output_path = results_dir / f"rut_{img_path.name}"
        cv2.imwrite(str(output_path), vis_img)
    
    # Preparar resultados para JSON (evitar referencias circulares)
    json_results = {}
    for detector_name, detector_results in results.items():
        json_results[detector_name] = []
        for result_item in detector_results:
            # Crear una copia simplificada del resultado
            simplified_result = {
                "image": result_item["image"],
                "present": bool(result_item["result"]["present"]),  # Convertir a bool nativo de Python
                "confidence": float(result_item["result"]["confidence"])
            }
            
            # Añadir metadatos específicos si existen
            if "metadata" in result_item["result"]:
                simplified_result["metadata"] = {}
                if detector_name == "fondo_blanco" and "porcentaje_blanco" in result_item["result"]["metadata"]:
                    simplified_result["metadata"]["porcentaje_blanco"] = float(result_item["result"]["metadata"]["porcentaje_blanco"])
            
            # Añadir RUT si existe
            if detector_name == "rut" and result_item["result"]["present"] and "rut" in result_item["result"]:
                simplified_result["rut"] = str(result_item["result"]["rut"])
                
            json_results[detector_name].append(simplified_result)
    
    # Guardar resultados en JSON
    with open(results_dir / "detector_results.json", "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, default=str)  # Usar default=str para convertir tipos no serializables
    
    print(f"\nResultados guardados en: {results_dir}")
    return results

def test_validator():
    """
    Prueba el sistema de validación completo con las imágenes de prueba.
    """
    # Directorios
    test_dir = Path("/home/ubuntu/detector_imagenes/test_images")
    results_dir = Path("/home/ubuntu/detector_imagenes/validation_results")
    results_dir.mkdir(exist_ok=True)
    
    # Inicializar validador
    validator = ImageValidator()
    
    print("\n=== Prueba del Sistema de Validación ===")
    
    # Validar cada imagen
    for img_path in test_dir.glob("*.jpg"):
        print(f"\nValidando imagen: {img_path.name}")
        
        # Validar imagen
        results = validator.validate_image(str(img_path))
        
        # Mostrar resultados
        for feature, present in results.items():
            if feature not in ['confidence', 'metadata']:
                print(f"  {feature}: {'Sí' if present else 'No'} (Confianza: {results['confidence'][feature]:.2f})")
        
        # Guardar informe
        report_path = validator.save_validation_report(str(img_path), results, str(results_dir))
        print(f"  Informe guardado en: {report_path}")
    
    print(f"\nInformes de validación guardados en: {results_dir}")

def evaluate_performance(detector_results):
    """
    Evalúa el rendimiento de los detectores basado en los resultados de las pruebas.
    
    Args:
        detector_results: Resultados de las pruebas de los detectores
    """
    # Definir ground truth (verdad fundamental) para cada imagen
    # Esto normalmente vendría de anotaciones manuales, pero para este ejemplo
    # lo definimos basado en los nombres de archivo
    ground_truth = {
        "fondo_blanco_1.jpg": {"fondo_blanco": True, "persona": False, "cedula": False, "rut": False},
        "fondo_blanco_2.jpg": {"fondo_blanco": True, "persona": False, "cedula": False, "rut": False},
        "no_fondo_blanco_1.jpg": {"fondo_blanco": False, "persona": False, "cedula": False, "rut": False},
        "no_fondo_blanco_2.jpg": {"fondo_blanco": False, "persona": False, "cedula": False, "rut": False},
        "cedula_1.jpg": {"fondo_blanco": True, "persona": False, "cedula": True, "rut": True},
        "cedula_2.jpg": {"fondo_blanco": True, "persona": False, "cedula": True, "rut": True},
        "rut_1.jpg": {"fondo_blanco": True, "persona": False, "cedula": False, "rut": True},
        "rut_2.jpg": {"fondo_blanco": True, "persona": False, "cedula": False, "rut": True},
        "persona_1.jpg": {"fondo_blanco": False, "persona": True, "cedula": False, "rut": False},
        "persona_2.jpg": {"fondo_blanco": False, "persona": True, "cedula": False, "rut": False},
        "persona_con_cedula.jpg": {"fondo_blanco": False, "persona": True, "cedula": True, "rut": True}
    }
    
    # Calcular métricas para cada detector
    metrics = {}
    
    for detector_name in ["fondo_blanco", "persona", "cedula", "rut"]:
        # Inicializar contadores
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        # Evaluar cada imagen
        for result_item in detector_results[detector_name]:
            image_name = result_item["image"]
            if image_name in ground_truth:
                # Obtener predicción y verdad
                prediction = result_item["result"]["present"]
                truth = ground_truth[image_name][detector_name]
                
                # Actualizar contadores
                if prediction and truth:
                    true_positives += 1
                elif prediction and not truth:
                    false_positives += 1
                elif not prediction and not truth:
                    true_negatives += 1
                elif not prediction and truth:
                    false_negatives += 1
        
        # Calcular métricas
        total = true_positives + false_positives + true_negatives + false_negatives
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Guardar métricas
        metrics[detector_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives
        }
    
    # Mostrar resultados
    print("\n=== Evaluación de Rendimiento ===")
    for detector_name, detector_metrics in metrics.items():
        print(f"\nDetector: {detector_name}")
        print(f"  Exactitud (Accuracy): {detector_metrics['accuracy']:.2f}")
        print(f"  Precisión (Precision): {detector_metrics['precision']:.2f}")
        print(f"  Exhaustividad (Recall): {detector_metrics['recall']:.2f}")
        print(f"  Puntuación F1 (F1 Score): {detector_metrics['f1_score']:.2f}")
        print(f"  Verdaderos Positivos: {detector_metrics['true_positives']}")
        print(f"  Falsos Positivos: {detector_metrics['false_positives']}")
        print(f"  Verdaderos Negativos: {detector_metrics['true_negatives']}")
        print(f"  Falsos Negativos: {detector_metrics['false_negatives']}")
    
    # Guardar métricas en JSON
    results_dir = Path("/home/ubuntu/detector_imagenes/test_results")
    with open(results_dir / "performance_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMétricas de rendimiento guardadas en: {results_dir / 'performance_metrics.json'}")
    
    return metrics

if __name__ == "__main__":
    # Probar detectores individuales
    detector_results = test_detectors()
    
    # Probar sistema de validación
    test_validator()
    
    # Evaluar rendimiento
    metrics = evaluate_performance(detector_results)
