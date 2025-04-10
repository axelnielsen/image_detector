"""
Detector de patrones oscuros de tipo falsos contadores de urgencia o escasez.
Identifica contadores, temporizadores o indicadores de escasez que generan presión artificial.
"""

import re
from typing import Dict, Any, List, Optional
import os
from pathlib import Path

from .base_detector import DarkPatternDetector


class FalseUrgencyDetector(DarkPatternDetector):
    """Detector de patrones de falsa urgencia o escasez."""
    
    def __init__(self):
        """Inicializa el detector de falsa urgencia o escasez."""
        super().__init__(
            name="Falsos contadores de urgencia o escasez",
            description="Patrón que utiliza contadores, temporizadores o indicadores de escasez para crear una sensación artificial de urgencia o escasez"
        )
        
        # Palabras clave relacionadas con urgencia
        self.urgency_keywords = [
            # Español
            "limitado", "limitada", "últimos", "últimas", "quedan", "queda", "pocas", "pocos",
            "agotándose", "agotado", "casi agotado", "escaso", "escasa", "exclusivo", "exclusiva",
            "no te lo pierdas", "no te la pierdas", "date prisa", "apresúrate", "corre",
            "por tiempo limitado", "oferta por tiempo limitado", "oferta especial",
            "termina pronto", "finaliza pronto", "acaba pronto", "solo hoy", "solo por hoy",
            "últimas horas", "últimos días", "últimas unidades", "últimas plazas",
            # Inglés
            "limited", "last", "few", "remaining", "remains", "left",
            "running out", "sold out", "almost gone", "scarce", "exclusive",
            "don't miss", "hurry", "rush", "limited time", "special offer",
            "ends soon", "today only", "last hours", "last days", "last units"
        ]
        
        # Patrones de texto que indican contadores o temporizadores
        self.countdown_patterns = [
            # Español
            r"(\d+)\s*(hora|horas|hr|hrs|minuto|minutos|min|mins|segundo|segundos|seg|segs)\s*(restante|restantes)",
            r"termina\s*en\s*(\d+)\s*(hora|horas|hr|hrs|minuto|minutos|min|mins|segundo|segundos|seg|segs)",
            r"finaliza\s*en\s*(\d+)\s*(hora|horas|hr|hrs|minuto|minutos|min|mins|segundo|segundos|seg|segs)",
            r"acaba\s*en\s*(\d+)\s*(hora|horas|hr|hrs|minuto|minutos|min|mins|segundo|segundos|seg|segs)",
            r"solo\s*(\d+)\s*(hora|horas|hr|hrs|minuto|minutos|min|mins|segundo|segundos|seg|segs)\s*más",
            r"(\d+):(\d+):(\d+)",  # Formato HH:MM:SS
            r"(\d+):(\d+)",  # Formato MM:SS
            # Inglés
            r"(\d+)\s*(hour|hours|hr|hrs|minute|minutes|min|mins|second|seconds|sec|secs)\s*(remaining|left)",
            r"ends\s*in\s*(\d+)\s*(hour|hours|hr|hrs|minute|minutes|min|mins|second|seconds|sec|secs)",
            r"finishes\s*in\s*(\d+)\s*(hour|hours|hr|hrs|minute|minutes|min|mins|second|seconds|sec|secs)",
            r"only\s*(\d+)\s*(hour|hours|hr|hrs|minute|minutes|min|mins|second|seconds|sec|secs)\s*more"
        ]
        
        # Patrones de texto que indican escasez
        self.scarcity_patterns = [
            # Español
            r"(solo|sólo|quedan|queda)\s*(\d+)\s*(en stock|disponibles|disponible|restantes|restante|unidades|unidad)",
            r"(\d+)%\s*(vendido|agotado)",
            r"(\d+)\s*(personas|clientes|usuarios)\s*(están viendo|han comprado|han reservado)",
            r"(alta|gran|mucha)\s*demanda",
            r"(más|mas)\s*(popular|vendido|demandado)",
            # Inglés
            r"(only|just)\s*(\d+)\s*(in stock|available|left|remaining|items|item)",
            r"(\d+)%\s*(sold|sold out|gone)",
            r"(\d+)\s*(people|customers|users)\s*(viewing|bought|booked|reserved)",
            r"(high|in)\s*demand",
            r"(most|best)\s*(popular|selling)"
        ]
    
    def detect(self, page_content: str, dom_structure: Dict[str, Any], 
               screenshot_path: str, url: str) -> List[Dict[str, Any]]:
        """
        Detecta patrones de falsa urgencia o escasez en una página.
        
        Args:
            page_content: Contenido HTML de la página
            dom_structure: Estructura DOM de la página
            screenshot_path: Ruta a la captura de pantalla de la página
            url: URL de la página
            
        Returns:
            List[Dict[str, Any]]: Lista de patrones de falsa urgencia o escasez detectados
        """
        detections = []
        
        # 1. Buscar contadores o temporizadores en el texto
        countdown_matches = self.search_text_patterns(page_content, self.countdown_patterns)
        
        for match in countdown_matches:
            # Calcular confianza basada en la presencia de palabras clave de urgencia
            urgency_keyword_count = sum(1 for word in self.urgency_keywords if word.lower() in match["context"].lower())
            confidence = self.calculate_confidence(urgency_keyword_count + 1, 0.8)
            
            if confidence >= self.confidence_threshold:
                detections.append({
                    "pattern_type": "false_urgency",
                    "evidence_type": "countdown",
                    "evidence": match,
                    "confidence": confidence,
                    "location": "Texto en página",
                    "screenshot": screenshot_path
                })
        
        # 2. Buscar indicadores de escasez en el texto
        scarcity_matches = self.search_text_patterns(page_content, self.scarcity_patterns)
        
        for match in scarcity_matches:
            # Calcular confianza basada en la presencia de palabras clave de urgencia
            urgency_keyword_count = sum(1 for word in self.urgency_keywords if word.lower() in match["context"].lower())
            confidence = self.calculate_confidence(urgency_keyword_count + 1, 0.8)
            
            if confidence >= self.confidence_threshold:
                detections.append({
                    "pattern_type": "false_urgency",
                    "evidence_type": "scarcity",
                    "evidence": match,
                    "confidence": confidence,
                    "location": "Texto en página",
                    "screenshot": screenshot_path
                })
        
        # 3. Buscar elementos visuales que puedan ser contadores o indicadores de escasez
        # Buscar elementos con clases o IDs que sugieran contadores
        countdown_elements = []
        
        def search_countdown_elements(node, path="body"):
            # Verificar si el nodo tiene clases o IDs que sugieren que es un contador
            is_countdown = False
            countdown_indicators = []
            
            # Palabras clave para identificar contadores en clases e IDs
            countdown_keywords = [
                "countdown", "timer", "clock", "counter", "remaining", "urgency",
                "contador", "temporizador", "reloj", "restante", "urgencia"
            ]
            
            # Verificar ID
            if node.get("id"):
                for keyword in countdown_keywords:
                    if keyword.lower() in node.get("id", "").lower():
                        is_countdown = True
                        countdown_indicators.append(f"id: {node.get('id')}")
            
            # Verificar clases
            if node.get("classes"):
                for cls in node.get("classes", []):
                    for keyword in countdown_keywords:
                        if keyword.lower() in cls.lower():
                            is_countdown = True
                            countdown_indicators.append(f"class: {cls}")
            
            # Verificar texto que parece un contador
            if node.get("text"):
                text = node.get("text", "")
                # Buscar formatos de tiempo como HH:MM:SS o MM:SS
                if re.search(r'\d+:\d+(:\d+)?', text):
                    is_countdown = True
                    countdown_indicators.append(f"text: {text}")
            
            # Si parece un contador, añadirlo a la lista
            if is_countdown:
                countdown_elements.append({
                    "node": node,
                    "path": path,
                    "indicators": countdown_indicators
                })
            
            # Buscar en nodos hijos
            if "children" in node:
                for i, child in enumerate(node["children"]):
                    child_path = f"{path} > {child.get('type', 'unknown')}[{i}]"
                    search_countdown_elements(child, child_path)
        
        # Iniciar búsqueda desde la raíz
        search_countdown_elements(dom_structure)
        
        # Analizar elementos de contador encontrados
        for element in countdown_elements:
            # Extraer texto del elemento
            element_text = ""
            
            def extract_element_text(node):
                nonlocal element_text
                if node.get("text"):
                    element_text += " " + node.get("text", "")
                
                if "children" in node:
                    for child in node["children"]:
                        extract_element_text(child)
            
            extract_element_text(element["node"])
            
            # Verificar si el texto contiene palabras clave de urgencia
            urgency_keyword_count = sum(1 for word in self.urgency_keywords if word.lower() in element_text.lower())
            
            # Calcular confianza
            confidence = self.calculate_confidence(len(element["indicators"]) + urgency_keyword_count, 0.85)
            
            if confidence >= self.confidence_threshold:
                detections.append({
                    "pattern_type": "false_urgency",
                    "evidence_type": "countdown_element",
                    "evidence": {
                        "path": element["path"],
                        "indicators": element["indicators"],
                        "text": element_text
                    },
                    "confidence": confidence,
                    "location": f"Contador en {element['path']}",
                    "screenshot": screenshot_path
                })
        
        # 4. Buscar elementos visuales que puedan ser indicadores de escasez
        # Buscar elementos con clases o IDs que sugieran indicadores de escasez
        scarcity_elements = []
        
        def search_scarcity_elements(node, path="body"):
            # Verificar si el nodo tiene clases o IDs que sugieren que es un indicador de escasez
            is_scarcity = False
            scarcity_indicators = []
            
            # Palabras clave para identificar indicadores de escasez en clases e IDs
            scarcity_keywords = [
                "stock", "inventory", "availability", "remaining", "left", "quantity",
                "popular", "trending", "demand", "hot", "selling", "sold",
                "existencia", "inventario", "disponibilidad", "restante", "cantidad",
                "popular", "tendencia", "demanda", "vendido"
            ]
            
            # Verificar ID
            if node.get("id"):
                for keyword in scarcity_keywords:
                    if keyword.lower() in node.get("id", "").lower():
                        is_scarcity = True
                        scarcity_indicators.append(f"id: {node.get('id')}")
            
            # Verificar clases
            if node.get("classes"):
                for cls in node.get("classes", []):
                    for keyword in scarcity_keywords:
                        if keyword.lower() in cls.lower():
                            is_scarcity = True
                            scarcity_indicators.append(f"class: {cls}")
            
            # Verificar texto que parece un indicador de escasez
            if node.get("text"):
                text = node.get("text", "")
                # Buscar patrones como "X disponibles" o "X% vendido"
                if re.search(r'\d+\s*(disponible|available|left|remaining|sold)', text, re.IGNORECASE):
                    is_scarcity = True
                    scarcity_indicators.append(f"text: {text}")
            
            # Si parece un indicador de escasez, añadirlo a la lista
            if is_scarcity:
                scarcity_elements.append({
                    "node": node,
                    "path": path,
                    "indicators": scarcity_indicators
                })
            
            # Buscar en nodos hijos
            if "children" in node:
                for i, child in enumerate(node["children"]):
                    child_path = f"{path} > {child.get('type', 'unknown')}[{i}]"
                    search_scarcity_elements(child, child_path)
        
        # Iniciar búsqueda desde la raíz
        search_scarcity_elements(dom_structure)
        
        # Analizar elementos de escasez encontrados
        for element in scarcity_elements:
            # Extraer texto del elemento
            element_text = ""
            
            def extract_element_text(node):
                nonlocal element_text
                if node.get("text"):
                    element_text += " " + node.get("text", "")
                
                if "children" in node:
                    for child in node["children"]:
                        extract_element_text(child)
            
            extract_element_text(element["node"])
            
            # Verificar si el texto contiene palabras clave de urgencia
            urgency_keyword_count = sum(1 for word in self.urgency_keywords if word.lower() in element_text.lower())
            
            # Calcular confianza
            confidence = self.calculate_confidence(len(element["indicators"]) + urgency_keyword_count, 0.85)
            
            if confidence >= self.confidence_threshold:
                detections.append({
                    "pattern_type": "false_urgency",
                    "evidence_type": "scarcity_element",
                    "evidence": {
                        "path": element["path"],
                        "indicators": element["indicators"],
                        "text": element_text
                    },
                    "confidence": confidence,
                    "location": f"Indicador de escasez en {element['path']}",
                    "screenshot": screenshot_path
                })
        
        return self.format_detection_result(detections)["detections"]
