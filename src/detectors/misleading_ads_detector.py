"""
Detector de patrones oscuros de tipo publicidad engañosa.
Identifica anuncios o contenido promocional que se presenta de manera confusa o engañosa.
"""

import re
from typing import Dict, Any, List, Optional
import os
from pathlib import Path

from .base_detector import DarkPatternDetector


class MisleadingAdsDetector(DarkPatternDetector):
    """Detector de patrones de publicidad engañosa."""
    
    def __init__(self):
        """Inicializa el detector de publicidad engañosa."""
        super().__init__(
            name="Publicidad engañosa",
            description="Patrón que presenta anuncios o contenido promocional de manera confusa o engañosa, haciéndolos parecer contenido orgánico o funcionalidades del sitio"
        )
        
        # Palabras clave relacionadas con publicidad
        self.ad_keywords = [
            # Español
            "anuncio", "publicidad", "patrocinado", "promocionado", "promoción",
            "oferta", "descuento", "especial", "limitado", "exclusivo",
            "colaboración", "contenido pagado", "recomendado", "sugerido",
            # Inglés
            "ad", "ads", "advert", "advertisement", "sponsored", "promoted", "promotion",
            "offer", "discount", "special", "limited", "exclusive",
            "partnership", "paid content", "recommended", "suggested"
        ]
        
        # Clases y IDs comunes para anuncios
        self.ad_classes_ids = [
            "ad", "ads", "advert", "advertisement", "banner", "promo", "promotion",
            "sponsored", "partner", "external", "commercial", "marketing",
            "anuncio", "publicidad", "promocion", "promoción", "oferta"
        ]
        
        # Patrones de URL que suelen indicar anuncios
        self.ad_url_patterns = [
            r'ad[sx]?\.', r'advert(ising)?\.', r'sponsor(ed)?\.', r'promo(tion)?\.', 
            r'banner\.', r'pop(up)?\.', r'click\.', r'track(ing)?\.', r'affiliate\.',
            r'campaign\.', r'market(ing)?\.', r'partner\.'
        ]
    
    def detect(self, page_content: str, dom_structure: Dict[str, Any], 
               screenshot_path: str, url: str) -> List[Dict[str, Any]]:
        """
        Detecta patrones de publicidad engañosa en una página.
        
        Args:
            page_content: Contenido HTML de la página
            dom_structure: Estructura DOM de la página
            screenshot_path: Ruta a la captura de pantalla de la página
            url: URL de la página
            
        Returns:
            List[Dict[str, Any]]: Lista de patrones de publicidad engañosa detectados
        """
        detections = []
        
        # 1. Buscar elementos que parezcan anuncios pero no estén claramente etiquetados
        # Buscar elementos con atributos que sugieren que son anuncios
        potential_ads = []
        
        def search_potential_ads(node, path="body"):
            # Verificar si el nodo tiene clases o IDs que sugieren que es un anuncio
            is_potential_ad = False
            ad_indicators = []
            
            # Verificar ID
            if node.get("id"):
                for keyword in self.ad_classes_ids:
                    if keyword.lower() in node.get("id", "").lower():
                        is_potential_ad = True
                        ad_indicators.append(f"id: {node.get('id')}")
            
            # Verificar clases
            if node.get("classes"):
                for cls in node.get("classes", []):
                    for keyword in self.ad_classes_ids:
                        if keyword.lower() in cls.lower():
                            is_potential_ad = True
                            ad_indicators.append(f"class: {cls}")
            
            # Verificar atributos
            if node.get("attributes"):
                # Verificar href en enlaces
                if "href" in node.get("attributes", {}) and node.get("type") == "A":
                    href = node.get("attributes", {}).get("href", "")
                    for pattern in self.ad_url_patterns:
                        if re.search(pattern, href, re.IGNORECASE):
                            is_potential_ad = True
                            ad_indicators.append(f"href: {href}")
                
                # Verificar src en imágenes
                if "src" in node.get("attributes", {}) and node.get("type") in ["IMG", "IFRAME"]:
                    src = node.get("attributes", {}).get("src", "")
                    for pattern in self.ad_url_patterns:
                        if re.search(pattern, src, re.IGNORECASE):
                            is_potential_ad = True
                            ad_indicators.append(f"src: {src}")
                
                # Verificar data-attributes relacionados con anuncios
                for attr, value in node.get("attributes", {}).items():
                    if attr.startswith("data-") and any(keyword.lower() in attr.lower() for keyword in self.ad_classes_ids):
                        is_potential_ad = True
                        ad_indicators.append(f"{attr}: {value}")
            
            # Si parece un anuncio, añadirlo a la lista
            if is_potential_ad:
                potential_ads.append({
                    "node": node,
                    "path": path,
                    "indicators": ad_indicators
                })
            
            # Buscar en nodos hijos
            if "children" in node:
                for i, child in enumerate(node["children"]):
                    child_path = f"{path} > {child.get('type', 'unknown')}[{i}]"
                    search_potential_ads(child, child_path)
        
        # Iniciar búsqueda desde la raíz
        search_potential_ads(dom_structure)
        
        # Analizar anuncios potenciales
        for ad in potential_ads:
            # Extraer texto del anuncio
            ad_text = ""
            
            def extract_ad_text(node):
                nonlocal ad_text
                if node.get("text"):
                    ad_text += " " + node.get("text", "")
                
                if "children" in node:
                    for child in node["children"]:
                        extract_ad_text(child)
            
            extract_ad_text(ad["node"])
            
            # Verificar si el anuncio está claramente etiquetado
            is_labeled = False
            for keyword in self.ad_keywords:
                if keyword.lower() in ad_text.lower():
                    is_labeled = True
                    break
            
            # Si no está claramente etiquetado, puede ser engañoso
            if not is_labeled:
                confidence = self.calculate_confidence(len(ad["indicators"]), 0.8)
                
                if confidence >= self.confidence_threshold:
                    detections.append({
                        "pattern_type": "misleading_ads",
                        "evidence_type": "unlabeled_ad",
                        "evidence": {
                            "path": ad["path"],
                            "indicators": ad["indicators"],
                            "text": ad_text
                        },
                        "confidence": confidence,
                        "location": f"Anuncio no etiquetado en {ad['path']}",
                        "screenshot": screenshot_path
                    })
        
        # 2. Buscar elementos que parezcan contenido orgánico pero sean anuncios
        # Buscar elementos que contengan palabras clave de anuncios en atributos ocultos
        native_ads = []
        
        def search_native_ads(node, path="body"):
            # Verificar si el nodo parece contenido orgánico pero tiene indicadores ocultos de anuncio
            is_content_like = node.get("type") in ["ARTICLE", "SECTION", "DIV"] and node.get("text")
            has_hidden_ad_indicators = False
            hidden_indicators = []
            
            if is_content_like:
                # Verificar atributos ocultos
                if node.get("attributes"):
                    for attr, value in node.get("attributes", {}).items():
                        if attr.startswith("data-") and any(keyword.lower() in value.lower() for keyword in self.ad_keywords):
                            has_hidden_ad_indicators = True
                            hidden_indicators.append(f"{attr}: {value}")
                
                # Verificar clases con nombres poco claros pero que podrían indicar anuncios
                if node.get("classes"):
                    for cls in node.get("classes", []):
                        if any(re.search(f"(^|[_-]){keyword}([_-]|$)", cls, re.IGNORECASE) for keyword in self.ad_classes_ids):
                            has_hidden_ad_indicators = True
                            hidden_indicators.append(f"class: {cls}")
            
            if is_content_like and has_hidden_ad_indicators:
                native_ads.append({
                    "node": node,
                    "path": path,
                    "indicators": hidden_indicators
                })
            
            # Buscar en nodos hijos
            if "children" in node:
                for i, child in enumerate(node["children"]):
                    child_path = f"{path} > {child.get('type', 'unknown')}[{i}]"
                    search_native_ads(child, child_path)
        
        # Iniciar búsqueda desde la raíz
        search_native_ads(dom_structure)
        
        # Analizar anuncios nativos
        for ad in native_ads:
            # Extraer texto del anuncio
            ad_text = ""
            
            def extract_ad_text(node):
                nonlocal ad_text
                if node.get("text"):
                    ad_text += " " + node.get("text", "")
                
                if "children" in node:
                    for child in node["children"]:
                        extract_ad_text(child)
            
            extract_ad_text(ad["node"])
            
            # Verificar si el anuncio está claramente etiquetado en el texto visible
            is_labeled = False
            for keyword in self.ad_keywords:
                if keyword.lower() in ad_text.lower():
                    is_labeled = True
                    break
            
            # Si no está claramente etiquetado, es un anuncio nativo engañoso
            if not is_labeled:
                confidence = self.calculate_confidence(len(ad["indicators"]), 0.85)
                
                if confidence >= self.confidence_threshold:
                    detections.append({
                        "pattern_type": "misleading_ads",
                        "evidence_type": "native_ad",
                        "evidence": {
                            "path": ad["path"],
                            "indicators": ad["indicators"],
                            "text": ad_text
                        },
                        "confidence": confidence,
                        "location": f"Anuncio nativo en {ad['path']}",
                        "screenshot": screenshot_path
                    })
        
        # 3. Buscar botones o enlaces que parezcan funcionalidades del sitio pero sean anuncios
        # Buscar elementos que parezcan botones o enlaces de navegación
        fake_ui_elements = []
        
        def search_fake_ui_elements(node, path="body"):
            # Verificar si el nodo parece un botón o enlace de navegación
            is_ui_element = node.get("type") in ["BUTTON", "A"] or (
                node.get("type") == "DIV" and 
                node.get("classes") and 
                any("button" in cls.lower() for cls in node.get("classes", []))
            )
            
            if is_ui_element:
                # Verificar si tiene atributos que sugieren que es un anuncio
                ad_indicators = []
                
                # Verificar href en enlaces
                if "attributes" in node and "href" in node.get("attributes", {}):
                    href = node.get("attributes", {}).get("href", "")
                    for pattern in self.ad_url_patterns:
                        if re.search(pattern, href, re.IGNORECASE):
                            ad_indicators.append(f"href: {href}")
                
                # Verificar data-attributes relacionados con anuncios
                if "attributes" in node:
                    for attr, value in node.get("attributes", {}).items():
                        if attr.startswith("data-") and any(keyword.lower() in attr.lower() for keyword in self.ad_classes_ids):
                            ad_indicators.append(f"{attr}: {value}")
                
                # Verificar clases con nombres poco claros pero que podrían indicar anuncios
                if node.get("classes"):
                    for cls in node.get("classes", []):
                        if any(re.search(f"(^|[_-]){keyword}([_-]|$)", cls, re.IGNORECASE) for keyword in self.ad_classes_ids):
                            ad_indicators.append(f"class: {cls}")
                
                if ad_indicators:
                    fake_ui_elements.append({
                        "node": node,
                        "path": path,
                        "indicators": ad_indicators,
                        "text": node.get("text", "")
                    })
            
            # Buscar en nodos hijos
            if "children" in node:
                for i, child in enumerate(node["children"]):
                    child_path = f"{path} > {child.get('type', 'unknown')}[{i}]"
                    search_fake_ui_elements(child, child_path)
        
        # Iniciar búsqueda desde la raíz
        search_fake_ui_elements(dom_structure)
        
        # Analizar elementos de UI falsos
        for element in fake_ui_elements:
            # Verificar si el elemento está claramente etiquetado como anuncio
            is_labeled = False
            for keyword in self.ad_keywords:
                if keyword.lower() in element["text"].lower():
                    is_labeled = True
                    break
            
            # Si no está claramente etiquetado, es un elemento de UI engañoso
            if not is_labeled:
                confidence = self.calculate_confidence(len(element["indicators"]), 0.9)
                
                if confidence >= self.confidence_threshold:
                    detections.append({
                        "pattern_type": "misleading_ads",
                        "evidence_type": "fake_ui",
                        "evidence": {
                            "path": element["path"],
                            "indicators": element["indicators"],
                            "text": element["text"]
                        },
                        "confidence": confidence,
                        "location": f"Elemento de UI engañoso en {element['path']}",
                        "screenshot": screenshot_path
                    })
        
        return self.format_detection_result(detections)["detections"]
