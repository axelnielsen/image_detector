"""
Detector de patrones oscuros de tipo suscripciones difíciles de cancelar.
Identifica interfaces que dificultan la cancelación de suscripciones o servicios.
"""

import re
from typing import Dict, Any, List, Optional
import os
from pathlib import Path

from .base_detector import DarkPatternDetector


class DifficultCancellationDetector(DarkPatternDetector):
    """Detector de patrones de suscripciones difíciles de cancelar."""
    
    def __init__(self):
        """Inicializa el detector de suscripciones difíciles de cancelar."""
        super().__init__(
            name="Suscripciones difíciles de cancelar",
            description="Patrón que dificulta la cancelación de suscripciones o servicios mediante interfaces confusas, procesos complejos o barreras adicionales"
        )
        
        # Palabras clave relacionadas con cancelación
        self.cancellation_keywords = [
            # Español
            "cancelar", "cancelación", "dar de baja", "baja", "desactivar", "desactivación",
            "terminar", "finalizar", "suspender", "suspensión", "eliminar", "eliminación",
            "cerrar", "cierre", "cuenta", "suscripción", "servicio", "membresía",
            # Inglés
            "cancel", "cancellation", "unsubscribe", "deactivate", "deactivation",
            "terminate", "termination", "end", "suspend", "suspension", "delete", "deletion",
            "close", "closure", "account", "subscription", "service", "membership"
        ]
        
        # Frases que indican procesos difíciles de cancelación
        self.difficult_cancellation_phrases = [
            # Español
            r"para\s+cancelar\s+(llame|llama|contacte|contacta|envíe|envía|escriba|escribe)",
            r"(cancelación|baja)\s+por\s+(teléfono|correo|email|escrito)",
            r"(cancelar|dar\s+de\s+baja)\s+en\s+persona",
            r"(cancelar|dar\s+de\s+baja)\s+enviando\s+(carta|solicitud)",
            r"(cancelar|dar\s+de\s+baja)\s+con\s+(\d+|un|una)\s+(día|días|semana|semanas|mes|meses)\s+de\s+antelación",
            r"(período|periodo)\s+de\s+(permanencia|compromiso)",
            r"(penalización|cargo|tarifa|comisión)\s+por\s+(cancelación|cancelar|baja|dar\s+de\s+baja)",
            r"no\s+es\s+posible\s+(cancelar|dar\s+de\s+baja)\s+online",
            r"(cancelación|baja)\s+sujeta\s+a\s+aprobación",
            # Inglés
            r"to\s+cancel\s+(call|contact|email|write)",
            r"(cancellation|unsubscribe)\s+by\s+(phone|mail|email|writing)",
            r"(cancel|unsubscribe)\s+in\s+person",
            r"(cancel|unsubscribe)\s+by\s+sending\s+(letter|request)",
            r"(cancel|unsubscribe)\s+with\s+(\d+|a|an)\s+(day|days|week|weeks|month|months)\s+notice",
            r"(minimum|commitment)\s+period",
            r"(penalty|fee|charge)\s+for\s+(cancellation|cancelling|unsubscribing)",
            r"not\s+possible\s+to\s+(cancel|unsubscribe)\s+online",
            r"(cancellation|unsubscription)\s+subject\s+to\s+approval"
        ]
    
    def detect(self, page_content: str, dom_structure: Dict[str, Any], 
               screenshot_path: str, url: str) -> List[Dict[str, Any]]:
        """
        Detecta patrones de suscripciones difíciles de cancelar en una página.
        
        Args:
            page_content: Contenido HTML de la página
            dom_structure: Estructura DOM de la página
            screenshot_path: Ruta a la captura de pantalla de la página
            url: URL de la página
            
        Returns:
            List[Dict[str, Any]]: Lista de patrones de suscripciones difíciles de cancelar detectados
        """
        detections = []
        
        # 1. Buscar frases que indiquen procesos difíciles de cancelación
        text_matches = self.search_text_patterns(page_content, self.difficult_cancellation_phrases)
        
        for match in text_matches:
            # Calcular confianza basada en la presencia de palabras clave de cancelación
            cancellation_keyword_count = sum(1 for word in self.cancellation_keywords if word.lower() in match["context"].lower())
            confidence = self.calculate_confidence(cancellation_keyword_count + 1, 0.85)
            
            if confidence >= self.confidence_threshold:
                detections.append({
                    "pattern_type": "difficult_cancellation",
                    "evidence_type": "text",
                    "evidence": match,
                    "confidence": confidence,
                    "location": "Texto en página",
                    "screenshot": screenshot_path
                })
        
        # 2. Buscar secciones relacionadas con cancelación y analizar su accesibilidad
        # Buscar elementos que contengan palabras clave de cancelación
        cancellation_sections = []
        
        def search_cancellation_sections(node, path="body", depth=0):
            # Verificar si el nodo actual contiene texto relacionado con cancelación
            if node.get("text"):
                text = node.get("text", "").lower()
                if any(keyword.lower() in text for keyword in self.cancellation_keywords):
                    cancellation_sections.append({
                        "node": node,
                        "path": path,
                        "depth": depth,
                        "text": text
                    })
            
            # Verificar atributos como ID o clase
            if node.get("id") and any(keyword.lower() in node.get("id", "").lower() for keyword in self.cancellation_keywords):
                cancellation_sections.append({
                    "node": node,
                    "path": path,
                    "depth": depth,
                    "id": node.get("id")
                })
            
            if node.get("classes"):
                for cls in node.get("classes", []):
                    if any(keyword.lower() in cls.lower() for keyword in self.cancellation_keywords):
                        cancellation_sections.append({
                            "node": node,
                            "path": path,
                            "depth": depth,
                            "class": cls
                        })
            
            # Buscar en nodos hijos
            if "children" in node:
                for i, child in enumerate(node["children"]):
                    child_path = f"{path} > {child.get('type', 'unknown')}[{i}]"
                    search_cancellation_sections(child, child_path, depth + 1)
        
        # Iniciar búsqueda desde la raíz
        search_cancellation_sections(dom_structure)
        
        # Analizar secciones de cancelación encontradas
        if cancellation_sections:
            # Verificar si las secciones de cancelación están muy anidadas (difíciles de encontrar)
            deep_sections = [section for section in cancellation_sections if section["depth"] > 3]
            
            if deep_sections and len(deep_sections) / len(cancellation_sections) > 0.5:
                confidence = self.calculate_confidence(len(deep_sections), 0.75)
                
                if confidence >= self.confidence_threshold:
                    detections.append({
                        "pattern_type": "difficult_cancellation",
                        "evidence_type": "deep_navigation",
                        "evidence": {
                            "sections": [
                                {
                                    "path": section["path"],
                                    "depth": section["depth"],
                                    "text": section.get("text", "N/A")
                                } 
                                for section in deep_sections
                            ]
                        },
                        "confidence": confidence,
                        "location": "Navegación profunda",
                        "screenshot": screenshot_path
                    })
            
            # Buscar enlaces o botones de cancelación
            cancellation_links = []
            for section in cancellation_sections:
                if "node" in section and "children" in section["node"]:
                    # Buscar enlaces o botones en los hijos
                    for child in section["node"].get("children", []):
                        if child.get("type") in ["A", "BUTTON"] and child.get("text"):
                            text = child.get("text", "").lower()
                            if any(keyword.lower() in text for keyword in self.cancellation_keywords):
                                cancellation_links.append({
                                    "text": text,
                                    "path": section["path"],
                                    "type": child.get("type")
                                })
            
            # Si no hay enlaces o botones de cancelación en secciones que hablan de cancelación,
            # puede ser indicio de que es difícil cancelar
            if cancellation_sections and not cancellation_links:
                confidence = self.calculate_confidence(len(cancellation_sections), 0.8)
                
                if confidence >= self.confidence_threshold:
                    detections.append({
                        "pattern_type": "difficult_cancellation",
                        "evidence_type": "no_cancellation_links",
                        "evidence": {
                            "sections": [
                                {
                                    "path": section["path"],
                                    "text": section.get("text", "N/A")
                                } 
                                for section in cancellation_sections
                            ]
                        },
                        "confidence": confidence,
                        "location": "Falta de enlaces de cancelación",
                        "screenshot": screenshot_path
                    })
        
        # 3. Buscar formularios complejos relacionados con cancelación
        # Buscar formularios que contengan palabras clave de cancelación
        forms = self.find_elements_by_attributes(dom_structure, {"type": "FORM"})
        
        for form in forms:
            form_text = ""
            
            # Extraer todo el texto del formulario
            def extract_form_text(node):
                nonlocal form_text
                if node.get("text"):
                    form_text += " " + node.get("text", "")
                
                if "children" in node:
                    for child in node["children"]:
                        extract_form_text(child)
            
            extract_form_text(form["node"])
            
            # Verificar si el formulario está relacionado con cancelación
            if any(keyword.lower() in form_text.lower() for keyword in self.cancellation_keywords):
                # Contar campos de entrada en el formulario
                input_fields = []
                
                def count_input_fields(node):
                    if node.get("type") in ["INPUT", "SELECT", "TEXTAREA"]:
                        input_fields.append(node)
                    
                    if "children" in node:
                        for child in node["children"]:
                            count_input_fields(child)
                
                count_input_fields(form["node"])
                
                # Si el formulario de cancelación tiene muchos campos, puede ser indicio de dificultad
                if len(input_fields) > 3:
                    confidence = self.calculate_confidence(len(input_fields), 0.8)
                    
                    if confidence >= self.confidence_threshold:
                        detections.append({
                            "pattern_type": "difficult_cancellation",
                            "evidence_type": "complex_form",
                            "evidence": {
                                "form_path": form["path"],
                                "form_text": form_text,
                                "input_count": len(input_fields)
                            },
                            "confidence": confidence,
                            "location": f"Formulario complejo en {form['path']}",
                            "screenshot": screenshot_path
                        })
        
        return self.format_detection_result(detections)["detections"]
