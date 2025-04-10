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
