"""
Módulo para la navegación automatizada de sitios web.
Utiliza Playwright para controlar un navegador headless.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext, ElementHandle
from PIL import Image


class WebCrawler:
    """Clase base para la navegación automatizada de sitios web."""
    
    def __init__(self, headless: bool = True, screenshots_dir: str = None, timeout: int = 30000):
        """
        Inicializa el navegador automatizado.
        
        Args:
            headless: Si True, el navegador se ejecuta en modo headless (sin interfaz gráfica)
            screenshots_dir: Directorio donde se guardarán las capturas de pantalla
            timeout: Tiempo máximo de espera para las operaciones en milisegundos
        """
        self.headless = headless
        self.timeout = timeout
        
        # Configurar directorio para capturas de pantalla
        if screenshots_dir:
            self.screenshots_dir = screenshots_dir
        else:
            self.screenshots_dir = os.path.join(os.getcwd(), 'data', 'screenshots')
        
        # Crear directorio si no existe
        os.makedirs(self.screenshots_dir, exist_ok=True)
        
        # Inicializar atributos que se configurarán más tarde
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.current_url = None
    
    def start(self) -> None:
        """Inicia el navegador y crea un nuevo contexto."""
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.context = self.browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )
        self.page = self.context.new_page()
        self.page.set_default_timeout(self.timeout)
    
    def stop(self) -> None:
        """Cierra el navegador y libera recursos."""
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
    
    def navigate(self, url: str) -> bool:
        """
        Navega a la URL especificada.
        
        Args:
            url: URL a la que navegar
            
        Returns:
            bool: True si la navegación fue exitosa, False en caso contrario
        """
        try:
            self.current_url = url
            response = self.page.goto(url, wait_until='networkidle', timeout=self.timeout)
            return response.ok
        except Exception as e:
            print(f"Error al navegar a {url}: {e}")
            return False
    
    def take_screenshot(self, name: str = None, full_page: bool = True) -> str:
        """
        Toma una captura de pantalla de la página actual.
        
        Args:
            name: Nombre para la captura de pantalla (sin extensión)
            full_page: Si True, captura toda la página, no solo la parte visible
            
        Returns:
            str: Ruta a la captura de pantalla guardada
        """
        if not name:
            # Generar nombre basado en la URL y timestamp
            domain = self.current_url.split('//')[1].split('/')[0].replace('.', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            name = f"{domain}_{timestamp}"
        
        # Asegurar que el nombre no contiene caracteres inválidos
        name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in name)
        
        # Crear ruta completa
        screenshot_path = os.path.join(self.screenshots_dir, f"{name}.png")
        
        # Tomar captura de pantalla
        self.page.screenshot(path=screenshot_path, full_page=full_page)
        
        return screenshot_path
    
    def take_element_screenshot(self, selector: str, name: str = None) -> Optional[str]:
        """
        Toma una captura de pantalla de un elemento específico.
        
        Args:
            selector: Selector CSS del elemento
            name: Nombre para la captura de pantalla (sin extensión)
            
        Returns:
            str: Ruta a la captura de pantalla guardada, o None si el elemento no se encuentra
        """
        try:
            element = self.page.query_selector(selector)
            if not element:
                print(f"Elemento no encontrado: {selector}")
                return None
            
            if not name:
                # Generar nombre basado en la URL, selector y timestamp
                domain = self.current_url.split('//')[1].split('/')[0].replace('.', '_')
                selector_short = selector.replace(' ', '_').replace('>', '_').replace(':', '_')[:20]
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                name = f"{domain}_{selector_short}_{timestamp}"
            
            # Asegurar que el nombre no contiene caracteres inválidos
            name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in name)
            
            # Crear ruta completa
            screenshot_path = os.path.join(self.screenshots_dir, f"{name}.png")
            
            # Tomar captura de pantalla del elemento
            element.screenshot(path=screenshot_path)
            
            return screenshot_path
        except Exception as e:
            print(f"Error al tomar captura del elemento {selector}: {e}")
            return None
    
    def get_page_content(self) -> str:
        """
        Obtiene el contenido HTML de la página actual.
        
        Returns:
            str: Contenido HTML de la página
        """
        return self.page.content()
    
    def get_dom_structure(self) -> Dict[str, Any]:
        """
        Obtiene la estructura DOM de la página actual en formato JSON.
        
        Returns:
            Dict[str, Any]: Estructura DOM simplificada
        """
        # Ejecutar JavaScript para obtener una representación simplificada del DOM
        dom_json = self.page.evaluate("""() => {
            function extractDomNode(node, maxDepth = 3, currentDepth = 0) {
                if (currentDepth > maxDepth) return { type: node.nodeName, truncated: true };
                
                const result = {
                    type: node.nodeName,
                    id: node.id || undefined,
                    classes: node.className ? node.className.split(' ').filter(c => c.length > 0) : undefined,
                    text: node.textContent ? (node.textContent.trim().substring(0, 100) || undefined) : undefined,
                    attributes: {},
                    children: []
                };
                
                // Extract attributes
                if (node.attributes) {
                    for (let i = 0; i < node.attributes.length; i++) {
                        const attr = node.attributes[i];
                        if (attr.name !== 'id' && attr.name !== 'class') {
                            result.attributes[attr.name] = attr.value;
                        }
                    }
                }
                
                // If no attributes, remove the empty object
                if (Object.keys(result.attributes).length === 0) {
                    delete result.attributes;
                }
                
                // Process children
                if (node.childNodes) {
                    for (let i = 0; i < node.childNodes.length; i++) {
                        const child = node.childNodes[i];
                        // Skip text nodes with only whitespace
                        if (child.nodeType === 3 && !child.textContent.trim()) continue;
                        
                        if (child.nodeType === 1) { // Element node
                            result.children.push(extractDomNode(child, maxDepth, currentDepth + 1));
                        }
                    }
                }
                
                // If no children, remove the empty array
                if (result.children.length === 0) {
                    delete result.children;
                }
                
                return result;
            }
            
            return extractDomNode(document.body);
        }""")
        
        return dom_json
    
    def click(self, selector: str, timeout: int = None) -> bool:
        """
        Hace clic en un elemento.
        
        Args:
            selector: Selector CSS del elemento
            timeout: Tiempo máximo de espera en milisegundos
            
        Returns:
            bool: True si el clic fue exitoso, False en caso contrario
        """
        try:
            if timeout is None:
                timeout = self.timeout
            self.page.click(selector, timeout=timeout)
            return True
        except Exception as e:
            print(f"Error al hacer clic en {selector}: {e}")
            return False
    
    def fill(self, selector: str, text: str) -> bool:
        """
        Rellena un campo de formulario.
        
        Args:
            selector: Selector CSS del elemento
            text: Texto a introducir
            
        Returns:
            bool: True si la operación fue exitosa, False en caso contrario
        """
        try:
            self.page.fill(selector, text)
            return True
        except Exception as e:
            print(f"Error al rellenar {selector}: {e}")
            return False
    
    def press(self, selector: str, key: str) -> bool:
        """
        Pulsa una tecla en un elemento.
        
        Args:
            selector: Selector CSS del elemento
            key: Tecla a pulsar (e.g., 'Enter', 'Tab', 'ArrowDown')
            
        Returns:
            bool: True si la operación fue exitosa, False en caso contrario
        """
        try:
            self.page.press(selector, key)
            return True
        except Exception as e:
            print(f"Error al pulsar {key} en {selector}: {e}")
            return False
    
    def wait_for_selector(self, selector: str, timeout: int = None) -> bool:
        """
        Espera a que un elemento esté presente en la página.
        
        Args:
            selector: Selector CSS del elemento
            timeout: Tiempo máximo de espera en milisegundos
            
        Returns:
            bool: True si el elemento apareció, False si se agotó el tiempo de espera
        """
        try:
            if timeout is None:
                timeout = self.timeout
            self.page.wait_for_selector(selector, timeout=timeout)
            return True
        except Exception as e:
            print(f"Tiempo de espera agotado para {selector}: {e}")
            return False
    
    def wait_for_navigation(self, timeout: int = None) -> bool:
        """
        Espera a que se complete una navegación.
        
        Args:
            timeout: Tiempo máximo de espera en milisegundos
            
        Returns:
            bool: True si la navegación se completó, False si se agotó el tiempo de espera
        """
        try:
            if timeout is None:
                timeout = self.timeout
            self.page.wait_for_load_state('networkidle', timeout=timeout)
            return True
        except Exception as e:
            print(f"Tiempo de espera agotado para navegación: {e}")
            return False
    
    def scroll_to_bottom(self, step: int = 250, delay: float = 0.1) -> None:
        """
        Desplaza la página hasta el final de forma gradual.
        
        Args:
            step: Píxeles a desplazar en cada paso
            delay: Tiempo de espera entre pasos en segundos
        """
        # Obtener altura de la página
        height = self.page.evaluate("() => document.body.scrollHeight")
        
        # Desplazar gradualmente
        current_position = 0
        while current_position < height:
            current_position += step
            self.page.evaluate(f"window.scrollTo(0, {current_position})")
            time.sleep(delay)
    
    def find_elements(self, selector: str) -> List[ElementHandle]:
        """
        Encuentra todos los elementos que coinciden con el selector.
        
        Args:
            selector: Selector CSS
            
        Returns:
            List[ElementHandle]: Lista de elementos encontrados
        """
        return self.page.query_selector_all(selector)
    
    def get_element_text(self, selector: str) -> Optional[str]:
        """
        Obtiene el texto de un elemento.
        
        Args:
            selector: Selector CSS del elemento
            
        Returns:
            str: Texto del elemento, o None si el elemento no se encuentra
        """
        try:
            element = self.page.query_selector(selector)
            if element:
                return element.text_content()
            return None
        except Exception as e:
            print(f"Error al obtener texto de {selector}: {e}")
            return None
    
    def get_element_attribute(self, selector: str, attribute: str) -> Optional[str]:
        """
        Obtiene el valor de un atributo de un elemento.
        
        Args:
            selector: Selector CSS del elemento
            attribute: Nombre del atributo
            
        Returns:
            str: Valor del atributo, o None si el elemento no se encuentra
        """
        try:
            element = self.page.query_selector(selector)
            if element:
                return element.get_attribute(attribute)
            return None
        except Exception as e:
            print(f"Error al obtener atributo {attribute} de {selector}: {e}")
            return None
    
    def get_cookies(self) -> List[Dict[str, Any]]:
        """
        Obtiene todas las cookies de la página actual.
        
        Returns:
            List[Dict[str, Any]]: Lista de cookies
        """
        return self.context.cookies()
    
    def clear_cookies(self) -> None:
        """Elimina todas las cookies."""
        self.context.clear_cookies()
    
    def execute_javascript(self, script: str) -> Any:
        """
        Ejecuta código JavaScript en la página.
        
        Args:
            script: Código JavaScript a ejecutar
            
        Returns:
            Any: Resultado de la ejecución
        """
        return self.page.evaluate(script)
    
    def __enter__(self):
        """Permite usar el crawler con el contexto 'with'."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cierra el navegador al salir del contexto 'with'."""
        self.stop()


class DarkPatternCrawler(WebCrawler):
    """Clase especializada para la detección de patrones oscuros."""
    
    def __init__(self, headless: bool = True, screenshots_dir: str = None, timeout: int = 30000):
        """
        Inicializa el crawler especializado en patrones oscuros.
        
        Args:
            headless: Si True, el navegador se ejecuta en modo headless (sin interfaz gráfica)
            screenshots_dir: Directorio donde se guardarán las capturas de pantalla
            timeout: Tiempo máximo de espera para las operaciones en milisegundos
        """
        super().__init__(headless, screenshots_dir, timeout)
        
        # Configurar directorio para evidencias
        self.evidence_dir = os.path.join(os.path.dirname(self.screenshots_dir), 'evidence')
        os.makedirs(self.evidence_dir, exist_ok=True)
    
    def analyze_page(self, url: str) -> Dict[str, Any]:
        """
        Analiza una página en busca de patrones oscuros.
        
        Args:
            url: URL de la página a analizar
            
        Returns:
            Dict[str, Any]: Resultados del análisis
        """
        # Navegar a la URL
        success = self.navigate(url)
        if not success:
            return {
                'url': url,
                'success': False,
                'error': 'No se pudo navegar a la página'
            }
        
        # Esperar a que la página cargue completamente
        self.wait_for_navigation()
        
        # Tomar captura de pantalla inicial
        screenshot_path = self.take_screenshot(name=f"initial_{url.split('//')[1].split('/')[0]}")
        
        # Recopilar información básica
        title = self.page.title()
        content = self.get_page_content()
        
        # Desplazarse por la página para cargar contenido dinámico
        self.scroll_to_bottom()
        
        # Tomar captura de pantalla después del desplazamiento
        full_screenshot_path = self.take_screenshot(name=f"full_{url.split('//')[1].split('/')[0]}")
        
        # Recopilar estructura DOM
        dom_structure = self.get_dom_structure()
        
        # Recopilar cookies
        cookies = self.get_cookies()
        
        return {
            'url': url,
            'success': True,
            'title': title,
            'screenshots': {
                'initial': screenshot_path,
                'full': full_screenshot_path
            },
            'dom_structure': dom_structure,
            'cookies': cookies,
            'html_length': len(content)
        }
    
    def save_evidence(self, evidence_type: str, data: Dict[str, Any], description: str = None) -> str:
        """
        Guarda evidencia de un patrón oscuro.
        
        Args:
            evidence_type: Tipo de patrón oscuro
            data: Datos de la evidencia
            description: Descripción de la evidencia
            
        Returns:
            str: Ruta al archivo de evidencia
        """
        # Generar nombre de archivo
        domain = self.current_url.split('//')[1].split('/')[0].replace('.', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{domain}_{evidence_type}_{timestamp}.json"
        
        # Crear ruta completa
        evidence_path = os.path.join(self.evidence_dir, filename)
        
        # Preparar datos
        evidence_data = {
            'url': self.current_url,
            'timestamp': datetime.now().isoformat(),
            'type': evidence_type,
            'description': description,
            'data': data
        }
        
        # Guardar en formato JSON
        import json
        with open(evidence_path, 'w', encoding='utf-8') as f:
            json.dump(evidence_data, f, indent=2, ensure_ascii=False)
        
        return evidence_path
