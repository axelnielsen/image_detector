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
