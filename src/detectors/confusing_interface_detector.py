"""
Detector de patrones oscuros de tipo interfaces confusas o botones engañosos.
Identifica interfaces que confunden al usuario o botones diseñados para engañar.
"""

import re
from typing import Dict, Any, List, Optional
import os
from pathlib import Path

from .base_detector import DarkPatternDetector


class ConfusingInterfaceDetector(DarkPatternDetector):
    """Detector de patrones de interfaces confusas o botones engañosos."""
    
    def __init__(self):
        """Inicializa el detector de interfaces confusas o botones engañosos."""
        super().__init__(
            name="Interfaces confusas o botones engañosos",
            description="Patrón que utiliza interfaces confusas o botones engañosos para inducir al usuario a realizar acciones no deseadas"
        )
        
        # Palabras clave para botones de acción primaria y secundaria
        self.primary_action_keywords = [
            # Español
            "aceptar", "continuar", "siguiente", "comprar", "pagar", "confirmar", 
            "suscribir", "registrar", "enviar", "guardar", "descargar",
            # Inglés
            "accept", "continue", "next", "buy", "pay", "confirm", 
            "subscribe", "register", "submit", "save", "download"
        ]
        
        self.secondary_action_keywords = [
            # Español
            "cancelar", "volver", "atrás", "salir", "cerrar", "rechazar", "no", 
            "omitir", "saltar", "más tarde", "ahora no", "no gracias",
            # Inglés
            "cancel", "back", "return", "exit", "close", "reject", "no", 
            "skip", "later", "not now", "no thanks"
        ]
        
        # Clases comunes para botones primarios y secundarios
        self.primary_button_classes = [
            "primary", "main", "cta", "action", "submit", "confirm", "buy", "pay",
            "principal", "principal-accion", "comprar", "pagar"
        ]
        
        self.secondary_button_classes = [
            "secondary", "cancel", "back", "return", "close", "reject", "skip",
            "secundario", "cancelar", "volver", "cerrar", "rechazar", "omitir"
        ]
    
    def detect(self, page_content: str, dom_structure: Dict[str, Any], 
               screenshot_path: str, url: str) -> List[Dict[str, Any]]:
        """
        Detecta patrones de interfaces confusas o botones engañosos en una página.
        
        Args:
            page_content: Contenido HTML de la página
            dom_structure: Estructura DOM de la página
            screenshot_path: Ruta a la captura de pantalla de la página
            url: URL de la página
            
        Returns:
            List[Dict[str, Any]]: Lista de patrones de interfaces confusas o botones engañosos detectados
        """
        # Verificar si la estructura DOM es válida
        if not dom_structure:
            print(f"Advertencia: Estructura DOM vacía o inválida para {url}")
            return []
            
        detections = []
        
        # 1. Buscar botones o enlaces con estilos engañosos
        # Buscar elementos que parezcan botones o enlaces
        buttons_and_links = []
        
        def search_buttons_and_links(node, path="body"):
            # Verificar que el nodo es un diccionario válido
            if not isinstance(node, dict):
                return
                
            # Verificar si el nodo es un botón o enlace
            node_type = node.get("type")
            if not node_type:
                return
                
            is_button_or_link = node_type in ["BUTTON", "A", "INPUT"] or (
                node_type == "DIV" and 
                node.get("classes") and 
                any(cls.lower() in ["button", "btn", "link", "boton", "enlace"] for cls in node.get("classes", []))
            )
            
            if is_button_or_link:
                # Extraer texto y atributos
                text = node.get("text", "")
                attributes = node.get("attributes", {})
                classes = node.get("classes", [])
                
                # Asegurar que classes es una lista
                if not isinstance(classes, list):
                    classes = []
                
                # Determinar si es un botón primario o secundario basado en el texto
                is_primary = any(keyword.lower() in text.lower() for keyword in self.primary_action_keywords) if text else False
                is_secondary = any(keyword.lower() in text.lower() for keyword in self.secondary_action_keywords) if text else False
                
                # Determinar si es un botón primario o secundario basado en las clases
                has_primary_class = any(cls.lower() in [c.lower() for c in classes] for cls in self.primary_button_classes) if classes else False
                has_secondary_class = any(cls.lower() in [c.lower() for c in classes] for cls in self.secondary_button_classes) if classes else False
                
                # Determinar tipo basado en atributos
                button_type = None
                if node_type == "INPUT" and attributes and "type" in attributes:
                    if attributes["type"] in ["submit", "button"]:
                        button_type = attributes["type"]
                
                buttons_and_links.append({
                    "node": node,
                    "path": path,
                    "text": text,
                    "classes": classes,
                    "attributes": attributes,
                    "is_primary_text": is_primary,
                    "is_secondary_text": is_secondary,
                    "has_primary_class": has_primary_class,
                    "has_secondary_class": has_secondary_class,
                    "button_type": button_type
                })
            
            # Buscar en nodos hijos
            children = node.get("children", [])
            if children and isinstance(children, list):
                for i, child in enumerate(children):
                    if isinstance(child, dict):
                        child_type = child.get("type", "unknown")
                        child_path = f"{path} > {child_type}[{i}]"
                        search_buttons_and_links(child, child_path)
        
        # Iniciar búsqueda desde la raíz con manejo de errores
        try:
            search_buttons_and_links(dom_structure)
        except Exception as e:
            print(f"Error al buscar botones y enlaces: {e}")
            return []
        
        # Analizar botones y enlaces para detectar inconsistencias
        for button in buttons_and_links:
            inconsistencies = []
            
            # Verificar inconsistencia entre texto y clase
            if button["is_primary_text"] and button["has_secondary_class"]:
                inconsistencies.append("Texto de acción primaria con clase de botón secundario")
            
            if button["is_secondary_text"] and button["has_primary_class"]:
                inconsistencies.append("Texto de acción secundaria con clase de botón primario")
            
            # Verificar si el botón de cancelar o rechazar tiene estilo visual prominente
            if button["is_secondary_text"] and button["has_primary_class"]:
                inconsistencies.append("Botón de cancelar/rechazar con estilo visual prominente")
            
            # Verificar si el botón de aceptar o confirmar tiene estilo visual poco prominente
            if button["is_primary_text"] and button["has_secondary_class"]:
                inconsistencies.append("Botón de aceptar/confirmar con estilo visual poco prominente")
            
            if inconsistencies:
                confidence = self.calculate_confidence(len(inconsistencies), 0.85)
                
                if confidence >= self.confidence_threshold:
                    detections.append({
                        "pattern_type": "confusing_interface",
                        "evidence_type": "misleading_button",
                        "evidence": {
                            "path": button["path"],
                            "text": button["text"],
                            "classes": button["classes"],
                            "inconsistencies": inconsistencies
                        },
                        "confidence": confidence,
                        "location": f"Botón engañoso en {button['path']}",
                        "screenshot": screenshot_path
                    })
        
        # 2. Buscar grupos de botones con jerarquía visual confusa
        # Agrupar botones que están cerca en el DOM
        button_groups = {}
        
        for i, button in enumerate(buttons_and_links):
            # Extraer el path del padre
            parent_path = " > ".join(button["path"].split(" > ")[:-1])
            
            if parent_path not in button_groups:
                button_groups[parent_path] = []
            
            button_groups[parent_path].append(button)
        
        # Analizar cada grupo de botones
        for parent_path, group in button_groups.items():
            if len(group) >= 2:  # Al menos dos botones en el grupo
                primary_buttons = [b for b in group if b["is_primary_text"] or b["has_primary_class"]]
                secondary_buttons = [b for b in group if b["is_secondary_text"] or b["has_secondary_class"]]
                
                # Verificar si hay múltiples botones primarios
                if len(primary_buttons) > 1:
                    confidence = self.calculate_confidence(len(primary_buttons), 0.75)
                    
                    if confidence >= self.confidence_threshold:
                        detections.append({
                            "pattern_type": "confusing_interface",
                            "evidence_type": "multiple_primary_buttons",
                            "evidence": {
                                "parent_path": parent_path,
                                "buttons": [
                                    {
                                        "path": b["path"],
                                        "text": b["text"],
                                        "classes": b["classes"]
                                    } 
                                    for b in primary_buttons
                                ]
                            },
                            "confidence": confidence,
                            "location": f"Múltiples botones primarios en {parent_path}",
                            "screenshot": screenshot_path
                        })
                
                # Verificar si hay botones primarios y secundarios con estilos similares
                if primary_buttons and secondary_buttons:
                    # Comparar clases para ver si son visualmente similares
                    similar_styles = False
                    
                    # Simplificación: si no hay clases distintivas, asumimos que son visualmente similares
                    primary_distinctive = any(b["has_primary_class"] for b in primary_buttons)
                    secondary_distinctive = any(b["has_secondary_class"] for b in secondary_buttons)
                    
                    if not (primary_distinctive and secondary_distinctive):
                        similar_styles = True
                    
                    if similar_styles:
                        confidence = self.calculate_confidence(len(primary_buttons) + len(secondary_buttons), 0.8)
                        
                        if confidence >= self.confidence_threshold:
                            detections.append({
                                "pattern_type": "confusing_interface",
                                "evidence_type": "similar_button_styles",
                                "evidence": {
                                    "parent_path": parent_path,
                                    "primary_buttons": [
                                        {
                                            "path": b["path"],
                                            "text": b["text"],
                                            "classes": b["classes"]
                                        } 
                                        for b in primary_buttons
                                    ],
                                    "secondary_buttons": [
                                        {
                                            "path": b["path"],
                                            "text": b["text"],
                                            "classes": b["classes"]
                                        } 
                                        for b in secondary_buttons
                                    ]
                                },
                                "confidence": confidence,
                                "location": f"Botones con estilos similares en {parent_path}",
                                "screenshot": screenshot_path
                            })
        
        # 3. Buscar elementos de interfaz que puedan ser confusos
        # Buscar elementos que parezcan controles de interfaz
        ui_elements = []
        
        def search_ui_elements(node, path="body"):
            # Verificar que el nodo es un diccionario válido
            if not isinstance(node, dict):
                return
                
            # Verificar si el nodo es un elemento de interfaz
            node_type = node.get("type")
            if not node_type:
                return
                
            is_ui_element = node_type in ["INPUT", "SELECT", "TEXTAREA", "LABEL", "FORM"]
            
            if is_ui_element:
                ui_elements.append({
                    "node": node,
                    "path": path,
                    "type": node_type,
                    "attributes": node.get("attributes", {}),
                    "classes": node.get("classes", []),
                    "text": node.get("text", "")
                })
            
            # Buscar en nodos hijos
            children = node.get("children", [])
            if children and isinstance(children, list):
                for i, child in enumerate(children):
                    if isinstance(child, dict):
                        child_type = child.get("type", "unknown")
                        child_path = f"{path} > {child_type}[{i}]"
                        search_ui_elements(child, child_path)
        
        # Iniciar búsqueda desde la raíz con manejo de errores
        try:
            search_ui_elements(dom_structure)
        except Exception as e:
            print(f"Error al buscar elementos de interfaz: {e}")
            return detections  # Devolver las detecciones que ya tenemos
        
        # Analizar elementos de interfaz para detectar confusiones
        for element in ui_elements:
            confusing_aspects = []
            
            # Verificar si es un checkbox o radio sin label claro
            if element["type"] == "INPUT" and element["attributes"].get("type") in ["checkbox", "radio"]:
                # Buscar label asociado
                has_label = False
                
                # Verificar si tiene ID
                if "id" in element["attributes"]:
                    element_id = element["attributes"]["id"]
                    
                    # Buscar label con atributo "for" que coincida con el ID
                    for ui_el in ui_elements:
                        if ui_el["type"] == "LABEL" and ui_el["attributes"].get("for") == element_id:
                            has_label = True
                            break
                
                if not has_label:
                    confusing_aspects.append("Checkbox o radio sin label claro")
            
            # Verificar si es un input con placeholder pero sin label
            if element["type"] == "INPUT" and "placeholder" in element["attributes"]:
                # Buscar label asociado
                has_label = False
                
                # Verificar si tiene ID
                if "id" in element["attributes"]:
                    element_id = element["attributes"]["id"]
                    
                    # Buscar label con atributo "for" que coincida con el ID
                    for ui_el in ui_elements:
                        if ui_el["type"] == "LABEL" and ui_el["attributes"].get("for") == element_id:
                            has_label = True
                            break
                
                if not has_label:
                    confusing_aspects.append("Input con placeholder pero sin label")
            
            # Verificar si es un formulario sin botón de cancelar claro
            if element["type"] == "FORM":
                has_submit = False
                has_cancel = False
                
                # Buscar botones dentro del formulario
                form_path = element["path"]
                
                for button in buttons_and_links:
                    if button["path"].startswith(form_path):
                        if button["is_primary_text"] or button["has_primary_class"] or button["button_type"] == "submit":
                            has_submit = True
                        
                        if button["is_secondary_text"] or button["has_secondary_class"]:
                            has_cancel = True
                
                if has_submit and not has_cancel:
                    confusing_aspects.append("Formulario sin botón de cancelar claro")
            
            if confusing_aspects:
                confidence = self.calculate_confidence(len(confusing_aspects), 0.7)
                
                if confidence >= self.confidence_threshold:
                    detections.append({
                        "pattern_type": "confusing_interface",
                        "evidence_type": "confusing_ui_element",
                        "evidence": {
                            "path": element["path"],
                            "type": element["type"],
                            "confusing_aspects": confusing_aspects
                        },
                        "confidence": confidence,
                        "location": f"Elemento de interfaz confuso en {element['path']}",
                        "screenshot": screenshot_path
                    })
        
        return detections
