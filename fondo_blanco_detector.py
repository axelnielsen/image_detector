"""
Detector de Fondo Blanco

Este módulo implementa un detector para identificar si una imagen tiene fondo blanco.
Utiliza análisis de histograma y umbralización para determinar el porcentaje de píxeles
blancos en la imagen.

Autor: Sistema de Detección de Imágenes
Fecha: Abril 2025
"""

import cv2
import numpy as np
import os
from pathlib import Path

class FondoBlancoDetector:
    """
    Detector para identificar si una imagen tiene fondo blanco.
    Utiliza análisis de histograma y umbralización para determinar
    el porcentaje de píxeles blancos en la imagen.
    """
    
    def __init__(self, threshold=0.85, white_threshold=230):
        """
        Inicializa el detector de fondo blanco.
        
        Args:
            threshold (float): Umbral de porcentaje de píxeles blancos para considerar fondo blanco.
            white_threshold (int): Valor de umbral para considerar un píxel como blanco (0-255).
        """
        self.threshold = threshold
        self.white_threshold = white_threshold
    
    def detect(self, image):
        """
        Detecta si la imagen tiene fondo blanco.
        
        Args:
            image: Imagen en formato NumPy array (BGR).
            
        Returns:
            dict: Resultado de la detección con formato:
                {
                    "present": bool,
                    "confidence": float,
                    "metadata": {
                        "porcentaje_blanco": float
                    }
                }
        """
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calcular el porcentaje de píxeles blancos
        white_pixels = np.sum(gray >= self.white_threshold)
        total_pixels = gray.size
        white_percentage = white_pixels / total_pixels
        
        # Determinar si tiene fondo blanco basado en el umbral
        is_white_background = white_percentage >= self.threshold
        
        # Calcular confianza basada en la distancia al umbral
        if is_white_background:
            confidence = min(1.0, white_percentage / self.threshold)
        else:
            confidence = min(1.0, (1.0 - white_percentage) / (1.0 - self.threshold))
        
        return {
            "present": is_white_background,
            "confidence": float(confidence),
            "metadata": {
                "porcentaje_blanco": float(white_percentage * 100)
            }
        }
    
    def visualize(self, image, result):
        """
        Genera una visualización del resultado de la detección.
        
        Args:
            image: Imagen original en formato NumPy array (BGR).
            result: Resultado de la detección.
            
        Returns:
            image: Imagen con visualización del resultado.
        """
        # Crear una copia de la imagen
        vis_image = image.copy()
        
        # Añadir texto con el resultado
        text = f"Fondo Blanco: {'Sí' if result['present'] else 'No'}"
        confidence_text = f"Confianza: {result['confidence']:.2f}"
        percentage_text = f"% Blanco: {result['metadata']['porcentaje_blanco']:.1f}%"
        
        # Definir color (verde si es positivo, rojo si es negativo)
        color = (0, 255, 0) if result['present'] else (0, 0, 255)
        
        # Añadir textos a la imagen
        cv2.putText(vis_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(vis_image, confidence_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(vis_image, percentage_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return vis_image
