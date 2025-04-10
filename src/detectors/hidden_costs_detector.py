"""
Detector de patrones oscuros de tipo cargos ocultos.
Identifica costos adicionales que se revelan tarde en el proceso de compra.
"""

import re
from typing import Dict, Any, List, Optional
import os
from pathlib import Path

from .base_detector import DarkPatternDetector


class HiddenCostsDetector(DarkPatternDetector):
    """Detector de patrones de cargos ocultos."""
    
    def __init__(self):
        """Inicializa el detector de cargos ocultos."""
        super().__init__(
            name="Cargos ocultos",
            description="Patrón que oculta costos adicionales hasta etapas avanzadas del proceso de compra"
        )
        
        # Palabras clave relacionadas con cargos adicionales
        self.cost_keywords = [
            # Español
            "cargo", "cargos", "tarifa", "tarifas", "comisión", "comisiones",
            "adicional", "adicionales", "extra", "extras", "suplemento", "suplementos",
            "impuesto", "impuestos", "tasa", "tasas", "IVA", "IGIC", "IRPF",
            "envío", "envíos", "entrega", "entregas", "manipulación", "manipulado",
            "procesamiento", "proceso", "gestión", "administración", "servicio", "servicios",
            "obligatorio", "obligatorios", "requerido", "requeridos",
            "seguro", "seguros", "garantía", "garantías", "protección",
            "mantenimiento", "suscripción", "suscripciones", "membresía", "membresías",
            # Inglés
            "fee", "fees", "charge", "charges", "surcharge", "surcharges",
            "additional", "extra", "supplement", "supplements",
            "tax", "taxes", "VAT", "GST", "HST",
            "shipping", "delivery", "handling", "processing", "service",
            "mandatory", "required", "compulsory",
            "insurance", "warranty", "protection",
            "maintenance", "subscription", "membership"
        ]
        
        # Patrones de texto que pueden indicar cargos ocultos
        self.hidden_cost_patterns = [
            # Español
            r"(cargo|tarifa|comisión|tasa)s?\s+(adicional|extra)e?s?",
            r"(cargo|tarifa|comisión|tasa)s?\s+de\s+(envío|entrega|manipulación|procesamiento|servicio)",
            r"(impuesto|IVA|IGIC|IRPF)s?\s+no\s+incluido",
            r"(más|no\s+incluye)\s+(gastos|costes)\s+de\s+(envío|entrega)",
            r"(se\s+añadirá|se\s+aplicará|se\s+cobrará)\s+(un|una|el|la)\s+(cargo|tarifa|comisión|impuesto)",
            r"(más|plus)\s+(IVA|impuestos|tasas)",
            r"(no\s+incluye|excluye)\s+(IVA|impuestos|tasas)",
            r"(cargo|tarifa|comisión)\s+por\s+(transacción|procesamiento|pago)",
            # Inglés
            r"(fee|charge|surcharge)s?\s+(additional|extra)",
            r"(fee|charge|surcharge)s?\s+for\s+(shipping|delivery|handling|processing|service)",
            r"(tax|VAT|GST|HST)e?s?\s+not\s+included",
            r"(plus|excluding)\s+(shipping|delivery)\s+(cost|fee|charge)s?",
            r"(will\s+be\s+added|will\s+apply|will\s+be\s+charged)\s+(a|an|the)\s+(fee|charge|surcharge|tax)",
            r"(plus|additional)\s+(VAT|tax|taxes)",
            r"(not\s+including|excludes)\s+(VAT|tax|taxes)",
            r"(fee|charge|surcharge)\s+for\s+(transaction|processing|payment)"
        ]
    
    def detect(self, page_content: str, dom_structure: Dict[str, Any], 
               screenshot_path: str, url: str) -> List[Dict[str, Any]]:
        """
        Detecta patrones de cargos ocultos en una página.
        
        Args:
            page_content: Contenido HTML de la página
            dom_structure: Estructura DOM de la página
            screenshot_path: Ruta a la captura de pantalla de la página
            url: URL de la página
            
        Returns:
            List[Dict[str, Any]]: Lista de patrones de cargos ocultos detectados
        """
        detections = []
        
        # 1. Buscar patrones de texto que indiquen cargos ocultos
        text_matches = self.search_text_patterns(page_content, self.hidden_cost_patterns)
        
        for match in text_matches:
            # Calcular confianza basada en la presencia de palabras clave de costos
            cost_keyword_count = sum(1 for word in self.cost_keywords if word.lower() in match["context"].lower())
            confidence = self.calculate_confidence(cost_keyword_count + 1, 0.8)
            
            if confidence >= self.confidence_threshold:
                detections.append({
                    "pattern_type": "hidden_costs",
                    "evidence_type": "text",
                    "evidence": match,
                    "confidence": confidence,
                    "location": "Texto en página",
                    "screenshot": screenshot_path
                })
        
        # 2. Buscar elementos que puedan contener información sobre cargos
        # Buscar elementos con texto relacionado con precios o costos
        price_elements = []
        
        def search_price_nodes(node, path="body"):
            # Verificar si el nodo actual contiene texto relacionado con precios
            if node.get("text"):
                text = node.get("text", "").lower()
                
                # Buscar patrones de precio (€, $, etc.)
                price_pattern = r'(\d+[.,]\d+|\d+)\s*[€$£¥]|[€$£¥]\s*(\d+[.,]\d+|\d+)'
                if re.search(price_pattern, text):
                    price_elements.append({
                        "node": node,
                        "path": path,
                        "text": text
                    })
                
                # Buscar palabras clave de costos
                elif any(keyword.lower() in text for keyword in self.cost_keywords):
                    price_elements.append({
                        "node": node,
                        "path": path,
                        "text": text
                    })
            
            # Buscar en nodos hijos
            if "children" in node:
                for i, child in enumerate(node["children"]):
                    child_path = f"{path} > {child.get('type', 'unknown')}[{i}]"
                    search_price_nodes(child, child_path)
        
        # Iniciar búsqueda desde la raíz
        search_price_nodes(dom_structure)
        
        # Analizar elementos de precio encontrados
        for element in price_elements:
            text = element["text"]
            
            # Verificar si el texto contiene indicios de cargos ocultos
            pattern_matches = any(re.search(pattern, text, re.IGNORECASE) for pattern in self.hidden_cost_patterns)
            
            # Verificar si el texto contiene palabras clave de costos
            cost_keyword_matches = [kw for kw in self.cost_keywords if kw.lower() in text.lower()]
            
            if pattern_matches or len(cost_keyword_matches) >= 2:
                confidence = self.calculate_confidence(
                    len(cost_keyword_matches) + (2 if pattern_matches else 0), 
                    0.85
                )
                
                if confidence >= self.confidence_threshold:
                    detections.append({
                        "pattern_type": "hidden_costs",
                        "evidence_type": "price_element",
                        "evidence": {
                            "text": text,
                            "path": element["path"],
                            "keywords": cost_keyword_matches
                        },
                        "confidence": confidence,
                        "location": f"Elemento de precio en {element['path']}",
                        "screenshot": screenshot_path
                    })
        
        # 3. Buscar elementos que aparezcan en secciones finales de checkout o carrito
        # Buscar elementos que contengan palabras clave de checkout
        checkout_keywords = ["checkout", "carrito", "cesta", "pago", "compra", "finalizar", "proceder", "cart", "basket", "payment", "purchase", "proceed"]
        
        checkout_sections = []
        
        def search_checkout_sections(node, path="body"):
            # Verificar si el nodo actual contiene texto relacionado con checkout
            if node.get("text"):
                text = node.get("text", "").lower()
                if any(keyword.lower() in text for keyword in checkout_keywords):
                    checkout_sections.append({
                        "node": node,
                        "path": path
                    })
            
            # Verificar atributos como ID o clase
            if node.get("id") and any(keyword.lower() in node.get("id", "").lower() for keyword in checkout_keywords):
                checkout_sections.append({
                    "node": node,
                    "path": path
                })
            
            if node.get("classes"):
                for cls in node.get("classes", []):
                    if any(keyword.lower() in cls.lower() for keyword in checkout_keywords):
                        checkout_sections.append({
                            "node": node,
                            "path": path
                        })
            
            # Buscar en nodos hijos
            if "children" in node:
                for i, child in enumerate(node["children"]):
                    child_path = f"{path} > {child.get('type', 'unknown')}[{i}]"
                    search_checkout_sections(child, child_path)
        
        # Iniciar búsqueda desde la raíz
        search_checkout_sections(dom_structure)
        
        # Analizar secciones de checkout encontradas
        for section in checkout_sections:
            # Buscar elementos de precio dentro de la sección de checkout
            section_prices = []
            
            def search_section_prices(node, path):
                if node.get("text"):
                    text = node.get("text", "").lower()
                    
                    # Buscar patrones de precio (€, $, etc.)
                    price_pattern = r'(\d+[.,]\d+|\d+)\s*[€$£¥]|[€$£¥]\s*(\d+[.,]\d+|\d+)'
                    if re.search(price_pattern, text):
                        section_prices.append({
                            "text": text,
                            "path": path
                        })
                    
                    # Buscar palabras clave de costos
                    elif any(keyword.lower() in text for keyword in self.cost_keywords):
                        section_prices.append({
                            "text": text,
                            "path": path
                        })
                
                # Buscar en nodos hijos
                if "children" in node:
                    for i, child in enumerate(node["children"]):
                        child_path = f"{path} > {child.get('type', 'unknown')}[{i}]"
                        search_section_prices(child, child_path)
            
            # Buscar precios en la sección de checkout
            search_section_prices(section["node"], section["path"])
            
            # Si hay múltiples elementos de precio en la sección de checkout, puede ser indicio de cargos ocultos
            if len(section_prices) >= 2:
                # Verificar si alguno contiene palabras clave de costos adicionales
                additional_costs = [
                    price for price in section_prices 
                    if any(kw.lower() in price["text"].lower() for kw in self.cost_keywords)
                ]
                
                if additional_costs:
                    confidence = self.calculate_confidence(len(additional_costs), 0.9)
                    
                    if confidence >= self.confidence_threshold:
                        detections.append({
                            "pattern_type": "hidden_costs",
                            "evidence_type": "checkout_costs",
                            "evidence": {
                                "section_path": section["path"],
                                "price_elements": [price["text"] for price in section_prices],
                                "additional_costs": [cost["text"] for cost in additional_costs]
                            },
                            "confidence": confidence,
                            "location": f"Sección de checkout en {section['path']}",
                            "screenshot": screenshot_path
                        })
        
        return self.format_detection_result(detections)["detections"]
