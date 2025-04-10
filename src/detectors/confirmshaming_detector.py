"""
Detector de patrones oscuros de tipo confirmshaming.
Identifica textos y elementos que hacen sentir mal al usuario por rechazar una opción.
"""

import re
from typing import Dict, Any, List, Optional
import os
from pathlib import Path

from .base_detector import DarkPatternDetector


class ConfirmshamingDetector(DarkPatternDetector):
    """Detector de patrones de confirmshaming (avergonzar al usuario por rechazar)."""
    
    def __init__(self):
        """Inicializa el detector de confirmshaming."""
        super().__init__(
            name="Confirmshaming",
            description="Patrón que avergüenza o hace sentir culpable al usuario por rechazar una oferta o no realizar una acción"
        )
        
        # Patrones de texto comunes en confirmshaming
        self.text_patterns = [
            r"no,?\s*(gracias,?)?\s*no\s*(quiero|me\s*interesa|necesito)",
            r"no\s*(quiero|me\s*interesa)\s*(ahorrar|mejorar|beneficiarme)",
            r"prefiero\s*pagar\s*(más|el\s*precio\s*completo)",
            r"no\s*me\s*importa\s*(ahorrar|mi\s*privacidad|mi\s*seguridad)",
            r"no\s*necesito\s*(descuentos|ofertas|ahorros)",
            r"(continuar|seguir)\s*sin\s*(descuentos|protección|beneficios)",
            r"(renuncio|renunciar)\s*a\s*(ahorros|beneficios|descuentos)",
            r"no\s*quiero\s*estar\s*informado",
            r"prefiero\s*perderme\s*(ofertas|novedades|descuentos)",
            r"(entiendo|acepto)\s*(los\s*riesgos|perderme\s*ofertas)",
            r"no\s*me\s*importa\s*(perderme|mi\s*experiencia)",
            r"(seguir|continuar)\s*siendo\s*(un\s*novato|principiante)",
            r"no\s*quiero\s*mejorar",
            r"prefiero\s*no\s*recibir\s*ayuda",
            # Patrones en inglés (para sitios internacionales)
            r"no,?\s*(thanks,?)?\s*i\s*(don't|do\s*not)\s*(want|need)",
            r"i\s*(don't|do\s*not)\s*want\s*to\s*(save|improve|benefit)",
            r"i\s*prefer\s*to\s*pay\s*(more|full\s*price)",
            r"i\s*(don't|do\s*not)\s*care\s*about\s*(saving|my\s*privacy|security)",
            r"i\s*(don't|do\s*not)\s*need\s*(discounts|offers|savings)",
            r"(continue|proceed)\s*without\s*(discounts|protection|benefits)",
            r"(i\s*give\s*up|forfeit)\s*(savings|benefits|discounts)",
            r"i\s*(don't|do\s*not)\s*want\s*to\s*be\s*informed",
            r"i\s*prefer\s*to\s*miss\s*(offers|news|discounts)",
            r"i\s*(understand|accept)\s*(the\s*risks|missing\s*out)",
            r"i\s*(don't|do\s*not)\s*care\s*about\s*(missing|my\s*experience)",
            r"(continue|remain)\s*(a\s*novice|beginner)",
            r"i\s*(don't|do\s*not)\s*want\s*to\s*improve",
            r"i\s*prefer\s*not\s*to\s*receive\s*help"
        ]
        
        # Palabras negativas o culpabilizadoras comunes
        self.negative_words = [
            "perder", "perderse", "perderme", "perderás", "perderá", 
            "riesgo", "riesgos", "peligro", "peligroso",
            "arrepentir", "arrepentirás", "arrepentirá",
            "lamentar", "lamentarás", "lamentará",
            "error", "equivocación", "equivocado",
            "peor", "malo", "mala", "negativo",
            "desaprovechar", "desperdiciar", "desaprovecharás",
            "renunciar", "renuncias", "renunciarás",
            "rechazar", "rechazas", "rechazarás",
            "ignorar", "ignoras", "ignorarás",
            "sin protección", "desprotegido", "vulnerable",
            "inseguro", "insegura", "peligroso", "peligrosa",
            # Palabras en inglés
            "miss", "missing", "lose", "losing", "lost",
            "risk", "risks", "danger", "dangerous",
            "regret", "sorry", "unfortunate",
            "mistake", "error", "wrong",
            "worse", "bad", "negative",
            "waste", "wasting", "squander",
            "give up", "forfeit", "surrender",
            "reject", "decline", "refuse",
            "ignore", "overlook", "disregard",
            "unprotected", "vulnerable", "exposed",
            "unsafe", "insecure", "risky"
        ]
    
    def detect(self, page_content: str, dom_structure: Dict[str, Any], 
               screenshot_path: str, url: str) -> List[Dict[str, Any]]:
        """
        Detecta patrones de confirmshaming en una página.
        
        Args:
            page_content: Contenido HTML de la página
            dom_structure: Estructura DOM de la página
            screenshot_path: Ruta a la captura de pantalla de la página
            url: URL de la página
            
        Returns:
            List[Dict[str, Any]]: Lista de patrones de confirmshaming detectados
        """
        detections = []
        
        # 1. Buscar patrones de texto en el contenido HTML
        text_matches = self.search_text_patterns(page_content, self.text_patterns)
        
        for match in text_matches:
            # Calcular confianza basada en la presencia de palabras negativas
            negative_word_count = sum(1 for word in self.negative_words if word.lower() in match["context"].lower())
            confidence = self.calculate_confidence(negative_word_count + 1, 0.8)
            
            if confidence >= self.confidence_threshold:
                detections.append({
                    "pattern_type": "confirmshaming",
                    "evidence_type": "text",
                    "evidence": match,
                    "confidence": confidence,
                    "location": "Texto en página",
                    "screenshot": screenshot_path
                })
        
        # 2. Buscar botones o enlaces de rechazo con texto negativo
        # Buscar elementos que parezcan botones o enlaces de rechazo
        decline_buttons = self.find_elements_by_attributes(
            dom_structure, 
            {"type": "BUTTON"}
        ) + self.find_elements_by_attributes(
            dom_structure, 
            {"type": "A"}
        )
        
        for button in decline_buttons:
            node = button["node"]
            text = node.get("text", "")
            
            if not text:
                continue
                
            # Verificar si el texto contiene patrones de confirmshaming
            negative_word_count = sum(1 for word in self.negative_words if word.lower() in text.lower())
            
            # Buscar patrones específicos en el texto del botón
            pattern_matches = any(re.search(pattern, text, re.IGNORECASE) for pattern in self.text_patterns)
            
            if negative_word_count > 0 or pattern_matches:
                confidence = self.calculate_confidence(negative_word_count + (1 if pattern_matches else 0), 0.9)
                
                if confidence >= self.confidence_threshold:
                    detections.append({
                        "pattern_type": "confirmshaming",
                        "evidence_type": "button_text",
                        "evidence": {
                            "text": text,
                            "path": button["path"],
                            "negative_words": [word for word in self.negative_words if word.lower() in text.lower()]
                        },
                        "confidence": confidence,
                        "location": f"Botón o enlace en {button['path']}",
                        "screenshot": screenshot_path
                    })
        
        # 3. Buscar formularios con opciones de rechazo negativas
        # Buscar elementos de formulario como checkboxes o radios
        form_elements = self.find_elements_by_attributes(
            dom_structure, 
            {"type": "INPUT"}
        ) + self.find_elements_by_attributes(
            dom_structure, 
            {"type": "LABEL"}
        )
        
        for element in form_elements:
            node = element["node"]
            text = node.get("text", "")
            
            if not text:
                continue
                
            # Verificar si el texto contiene patrones de confirmshaming
            negative_word_count = sum(1 for word in self.negative_words if word.lower() in text.lower())
            
            # Buscar patrones específicos en el texto del elemento
            pattern_matches = any(re.search(pattern, text, re.IGNORECASE) for pattern in self.text_patterns)
            
            if negative_word_count > 0 or pattern_matches:
                confidence = self.calculate_confidence(negative_word_count + (1 if pattern_matches else 0), 0.85)
                
                if confidence >= self.confidence_threshold:
                    detections.append({
                        "pattern_type": "confirmshaming",
                        "evidence_type": "form_element",
                        "evidence": {
                            "text": text,
                            "path": element["path"],
                            "negative_words": [word for word in self.negative_words if word.lower() in text.lower()]
                        },
                        "confidence": confidence,
                        "location": f"Elemento de formulario en {element['path']}",
                        "screenshot": screenshot_path
                    })
        
        return self.format_detection_result(detections)["detections"]
