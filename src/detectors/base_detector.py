"""
Módulo base para la detección de patrones oscuros.
Define la interfaz y funcionalidades comunes para todos los detectores.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import re
import json
import os
from pathlib import Path


class DarkPatternDetector(ABC):
    """Clase base abstracta para todos los detectores de patrones oscuros."""
    
    def __init__(self, name: str, description: str):
        """
        Inicializa el detector base.
        
        Args:
            name: Nombre del patrón oscuro
            description: Descripción del patrón oscuro
        """
        self.name = name
        self.description = description
        self.confidence_threshold = 0.7  # Umbral de confianza predeterminado
    
    @abstractmethod
    def detect(self, page_content: str, dom_structure: Dict[str, Any], 
               screenshot_path: str, url: str) -> List[Dict[str, Any]]:
        """
        Detecta patrones oscuros en una página.
        
        Args:
            page_content: Contenido HTML de la página
            dom_structure: Estructura DOM de la página
            screenshot_path: Ruta a la captura de pantalla de la página
            url: URL de la página
            
        Returns:
            List[Dict[str, Any]]: Lista de patrones oscuros detectados
        """
        pass
    
    def get_improvement_suggestion(self, pattern_type: str) -> str:
        """
        Genera una sugerencia de mejora para un patrón oscuro detectado.
        
        Args:
            pattern_type: Tipo específico de patrón oscuro
            
        Returns:
            str: Sugerencia de mejora
        """
        # Sugerencias genéricas por tipo de patrón
        suggestions = {
            "confirmshaming": "Utilice un lenguaje neutral para las opciones de rechazo. Evite hacer que los usuarios se sientan culpables por declinar.",
            "preselection": "Las opciones que implican costos adicionales o compartir datos no deberían estar preseleccionadas. Permita que los usuarios elijan activamente.",
            "hidden_costs": "Muestre todos los costos desde el principio del proceso. Evite añadir cargos sorpresa en las últimas etapas.",
            "difficult_cancellation": "Haga que el proceso de cancelación sea tan sencillo como el de suscripción. Proporcione un enlace directo a la cancelación.",
            "misleading_ads": "Distinga claramente entre contenido publicitario y contenido orgánico. Evite diseños que confundan anuncios con funcionalidades del sitio.",
            "false_urgency": "Utilice indicadores de urgencia solo cuando sean reales. Evite contadores falsos o mensajes de escasez fabricados.",
            "confusing_interface": "Diseñe interfaces claras con jerarquía visual adecuada. Asegúrese de que los botones de acción principal y secundaria sean visualmente distintos."
        }
        
        return suggestions.get(pattern_type, "Revise el diseño para asegurar que respeta la autonomía del usuario y proporciona información clara y honesta.")
    
    def format_detection_result(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Formatea los resultados de detección para el informe.
        
        Args:
            detections: Lista de patrones oscuros detectados
            
        Returns:
            Dict[str, Any]: Resultados formateados
        """
        if not detections:
            return {
                "pattern_name": self.name,
                "detected": False,
                "detections": []
            }
        
        # Añadir sugerencias de mejora a cada detección
        for detection in detections:
            if "improvement_suggestion" not in detection:
                detection["improvement_suggestion"] = self.get_improvement_suggestion(detection.get("pattern_type", ""))
        
        return {
            "pattern_name": self.name,
            "detected": True,
            "detections": detections
        }
    
    def search_text_patterns(self, text: str, patterns: List[str], 
                             context_chars: int = 50) -> List[Dict[str, Any]]:
        """
        Busca patrones de texto en el contenido.
        
        Args:
            text: Texto donde buscar
            patterns: Lista de patrones regex a buscar
            context_chars: Número de caracteres de contexto a incluir
            
        Returns:
            List[Dict[str, Any]]: Lista de coincidencias con contexto
        """
        results = []
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start_pos = max(0, match.start() - context_chars)
                end_pos = min(len(text), match.end() + context_chars)
                
                # Obtener contexto antes y después
                context = text[start_pos:end_pos]
                
                # Resaltar la coincidencia en el contexto
                match_in_context = context.replace(
                    match.group(0), 
                    f"**{match.group(0)}**"
                )
                
                results.append({
                    "match": match.group(0),
                    "context": match_in_context,
                    "position": {
                        "start": match.start(),
                        "end": match.end()
                    }
                })
        
        return results
    
    def find_elements_by_attributes(self, dom_structure: Dict[str, Any], 
                                   attribute_filters: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Busca elementos en el DOM que coincidan con los filtros de atributos.
        
        Args:
            dom_structure: Estructura DOM de la página
            attribute_filters: Diccionario de atributos y valores a buscar
            
        Returns:
            List[Dict[str, Any]]: Lista de elementos que coinciden
        """
        matches = []
        
        def search_node(node, path="body"):
            # Verificar si el nodo actual coincide con los filtros
            node_matches = True
            
            # Verificar tipo de nodo
            if "type" in attribute_filters and node.get("type", "").lower() != attribute_filters["type"].lower():
                node_matches = False
            
            # Verificar ID
            if "id" in attribute_filters and node.get("id", "") != attribute_filters["id"]:
                node_matches = False
            
            # Verificar clases
            if "class" in attribute_filters and node.get("classes"):
                if not any(c.lower() == attribute_filters["class"].lower() for c in node.get("classes", [])):
                    node_matches = False
            
            # Verificar texto
            if "text" in attribute_filters and node.get("text"):
                if attribute_filters["text"].lower() not in node.get("text", "").lower():
                    node_matches = False
            
            # Verificar atributos personalizados
            for attr_key, attr_value in attribute_filters.items():
                if attr_key not in ["type", "id", "class", "text"]:
                    if not node.get("attributes") or attr_key not in node.get("attributes") or node["attributes"][attr_key] != attr_value:
                        node_matches = False
            
            # Si coincide, añadir a los resultados
            if node_matches:
                matches.append({
                    "node": node,
                    "path": path
                })
            
            # Buscar en nodos hijos
            if "children" in node:
                for i, child in enumerate(node["children"]):
                    child_path = f"{path} > {child.get('type', 'unknown')}[{i}]"
                    search_node(child, child_path)
        
        # Iniciar búsqueda desde la raíz
        search_node(dom_structure)
        return matches
    
    def calculate_confidence(self, evidence_count: int, evidence_strength: float) -> float:
        """
        Calcula el nivel de confianza de una detección.
        
        Args:
            evidence_count: Número de evidencias encontradas
            evidence_strength: Fuerza de las evidencias (0.0-1.0)
            
        Returns:
            float: Nivel de confianza (0.0-1.0)
        """
        # Fórmula simple: más evidencias y más fuertes = mayor confianza
        base_confidence = min(0.5 + (evidence_count * 0.1), 0.9)
        return base_confidence * evidence_strength
