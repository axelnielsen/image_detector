"""
Detector de patrones oscuros de tipo preselección de opciones.
Identifica opciones preseleccionadas que pueden no ser beneficiosas para el usuario.
"""

import re
from typing import Dict, Any, List, Optional
import os
from pathlib import Path

from .base_detector import DarkPatternDetector


class PreselectionDetector(DarkPatternDetector):
    """Detector de patrones de preselección de opciones."""
    
    def __init__(self):
        """Inicializa el detector de preselección."""
        super().__init__(
            name="Preselección de opciones",
            description="Patrón que preselecciona opciones que pueden no ser beneficiosas para el usuario, como suscripciones, servicios adicionales o compartición de datos"
        )
        
        # Palabras clave relacionadas con opciones que suelen preseleccionarse
        self.preselection_keywords = [
            "newsletter", "boletín", "noticias", "ofertas", "promociones",
            "notificaciones", "avisos", "alertas", "actualizaciones",
            "compartir", "datos", "información", "marketing", "publicidad",
            "terceros", "socios", "afiliados", "comerciales",
            "suscripción", "suscribir", "premium", "adicional", "extra",
            "seguro", "garantía", "protección", "cobertura",
            "acepto", "autorizo", "consiento", "permito",
            # Palabras en inglés
            "newsletter", "bulletin", "news", "offers", "promotions",
            "notifications", "alerts", "updates",
            "share", "data", "information", "marketing", "advertising",
            "third party", "partners", "affiliates", "commercial",
            "subscription", "subscribe", "premium", "additional", "extra",
            "insurance", "warranty", "protection", "coverage",
            "accept", "authorize", "consent", "allow"
        ]
    
    def detect(self, page_content: str, dom_structure: Dict[str, Any], 
               screenshot_path: str, url: str) -> List[Dict[str, Any]]:
        """
        Detecta patrones de preselección en una página.
        
        Args:
            page_content: Contenido HTML de la página
            dom_structure: Estructura DOM de la página
            screenshot_path: Ruta a la captura de pantalla de la página
            url: URL de la página
            
        Returns:
            List[Dict[str, Any]]: Lista de patrones de preselección detectados
        """
        detections = []
        
        # 1. Buscar elementos de formulario preseleccionados
        # Buscar checkboxes, radios y selects
        checkboxes = self.find_elements_by_attributes(dom_structure, {"type": "INPUT", "attributes": {"type": "checkbox"}})
        radios = self.find_elements_by_attributes(dom_structure, {"type": "INPUT", "attributes": {"type": "radio"}})
        selects = self.find_elements_by_attributes(dom_structure, {"type": "SELECT"})
        
        # Analizar checkboxes
        for checkbox in checkboxes:
            node = checkbox["node"]
            attributes = node.get("attributes", {})
            
            # Verificar si está marcado por defecto
            is_checked = "checked" in attributes or attributes.get("checked") == "checked"
            
            if is_checked:
                # Buscar texto asociado (generalmente en un label cercano)
                associated_text = self._find_associated_text(dom_structure, checkbox["path"])
                
                if not associated_text:
                    continue
                
                # Verificar si el texto contiene palabras clave de preselección
                keyword_matches = [kw for kw in self.preselection_keywords if kw.lower() in associated_text.lower()]
                
                if keyword_matches:
                    confidence = self.calculate_confidence(len(keyword_matches), 0.9)
                    
                    if confidence >= self.confidence_threshold:
                        detections.append({
                            "pattern_type": "preselection",
                            "evidence_type": "checkbox",
                            "evidence": {
                                "text": associated_text,
                                "path": checkbox["path"],
                                "keywords": keyword_matches
                            },
                            "confidence": confidence,
                            "location": f"Checkbox en {checkbox['path']}",
                            "screenshot": screenshot_path
                        })
        
        # Analizar radios
        # Agrupar radios por nombre
        radio_groups = {}
        for radio in radios:
            node = radio["node"]
            attributes = node.get("attributes", {})
            name = attributes.get("name", "unknown")
            
            if name not in radio_groups:
                radio_groups[name] = []
            
            radio_groups[name].append({
                "node": node,
                "path": radio["path"],
                "checked": "checked" in attributes or attributes.get("checked") == "checked"
            })
        
        # Analizar cada grupo de radios
        for name, group in radio_groups.items():
            checked_radios = [r for r in group if r["checked"]]
            
            if checked_radios:
                # Hay un radio preseleccionado
                for checked_radio in checked_radios:
                    associated_text = self._find_associated_text(dom_structure, checked_radio["path"])
                    
                    if not associated_text:
                        continue
                    
                    # Verificar si el texto contiene palabras clave de preselección
                    keyword_matches = [kw for kw in self.preselection_keywords if kw.lower() in associated_text.lower()]
                    
                    if keyword_matches:
                        confidence = self.calculate_confidence(len(keyword_matches), 0.85)
                        
                        if confidence >= self.confidence_threshold:
                            detections.append({
                                "pattern_type": "preselection",
                                "evidence_type": "radio",
                                "evidence": {
                                    "text": associated_text,
                                    "path": checked_radio["path"],
                                    "keywords": keyword_matches
                                },
                                "confidence": confidence,
                                "location": f"Radio button en {checked_radio['path']}",
                                "screenshot": screenshot_path
                            })
        
        # Analizar selects
        for select in selects:
            node = select["node"]
            
            # Buscar opciones dentro del select
            options = []
            if "children" in node:
                for child in node["children"]:
                    if child.get("type") == "OPTION":
                        options.append(child)
            
            # Verificar si hay una opción seleccionada por defecto
            selected_options = [opt for opt in options if opt.get("attributes", {}).get("selected") == "selected"]
            
            if selected_options:
                for selected in selected_options:
                    option_text = selected.get("text", "")
                    
                    if not option_text:
                        continue
                    
                    # Verificar si el texto contiene palabras clave de preselección
                    keyword_matches = [kw for kw in self.preselection_keywords if kw.lower() in option_text.lower()]
                    
                    if keyword_matches:
                        confidence = self.calculate_confidence(len(keyword_matches), 0.8)
                        
                        if confidence >= self.confidence_threshold:
                            detections.append({
                                "pattern_type": "preselection",
                                "evidence_type": "select",
                                "evidence": {
                                    "text": option_text,
                                    "path": select["path"],
                                    "keywords": keyword_matches
                                },
                                "confidence": confidence,
                                "location": f"Select en {select['path']}",
                                "screenshot": screenshot_path
                            })
        
        # 2. Buscar patrones en el HTML que indiquen preselección
        # Buscar atributos checked y selected en el HTML
        checked_pattern = r'<input[^>]*\s+checked\s*[^>]*>'
        selected_pattern = r'<option[^>]*\s+selected\s*[^>]*>'
        
        checked_matches = re.finditer(checked_pattern, page_content)
        selected_matches = re.finditer(selected_pattern, page_content)
        
        # Analizar coincidencias de checked
        for match in checked_matches:
            match_text = match.group(0)
            
            # Extraer contexto alrededor del match
            start_pos = max(0, match.start() - 100)
            end_pos = min(len(page_content), match.end() + 100)
            context = page_content[start_pos:end_pos]
            
            # Verificar si el contexto contiene palabras clave de preselección
            keyword_matches = [kw for kw in self.preselection_keywords if kw.lower() in context.lower()]
            
            if keyword_matches:
                confidence = self.calculate_confidence(len(keyword_matches), 0.75)
                
                if confidence >= self.confidence_threshold:
                    detections.append({
                        "pattern_type": "preselection",
                        "evidence_type": "html_checked",
                        "evidence": {
                            "html": match_text,
                            "context": context,
                            "keywords": keyword_matches
                        },
                        "confidence": confidence,
                        "location": "Código HTML",
                        "screenshot": screenshot_path
                    })
        
        # Analizar coincidencias de selected
        for match in selected_matches:
            match_text = match.group(0)
            
            # Extraer contexto alrededor del match
            start_pos = max(0, match.start() - 100)
            end_pos = min(len(page_content), match.end() + 100)
            context = page_content[start_pos:end_pos]
            
            # Verificar si el contexto contiene palabras clave de preselección
            keyword_matches = [kw for kw in self.preselection_keywords if kw.lower() in context.lower()]
            
            if keyword_matches:
                confidence = self.calculate_confidence(len(keyword_matches), 0.75)
                
                if confidence >= self.confidence_threshold:
                    detections.append({
                        "pattern_type": "preselection",
                        "evidence_type": "html_selected",
                        "evidence": {
                            "html": match_text,
                            "context": context,
                            "keywords": keyword_matches
                        },
                        "confidence": confidence,
                        "location": "Código HTML",
                        "screenshot": screenshot_path
                    })
        
        return self.format_detection_result(detections)["detections"]
    
    def _find_associated_text(self, dom_structure: Dict[str, Any], element_path: str) -> Optional[str]:
        """
        Busca texto asociado a un elemento de formulario.
        
        Args:
            dom_structure: Estructura DOM de la página
            element_path: Ruta del elemento en el DOM
            
        Returns:
            Optional[str]: Texto asociado o None si no se encuentra
        """
        # Extraer ID del elemento
        element_id = None
        path_parts = element_path.split(" > ")
        for part in path_parts:
            if "[id=" in part:
                id_match = re.search(r'\[id=([^\]]+)\]', part)
                if id_match:
                    element_id = id_match.group(1)
        
        if element_id:
            # Buscar labels que apunten a este ID
            labels = self.find_elements_by_attributes(
                dom_structure, 
                {"type": "LABEL", "attributes": {"for": element_id}}
            )
            
            if labels:
                return labels[0]["node"].get("text", "")
        
        # Si no se encuentra por ID, buscar texto cercano
        # Esto es una aproximación simple; en un sistema real sería más complejo
        parent_path = " > ".join(element_path.split(" > ")[:-1])
        parent_elements = self.find_elements_by_attributes(
            dom_structure, 
            {"path": parent_path}
        )
        
        if parent_elements:
            parent = parent_elements[0]["node"]
            if "text" in parent and parent["text"]:
                return parent["text"]
        
        return None
