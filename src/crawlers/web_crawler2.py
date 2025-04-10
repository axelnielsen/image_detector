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
