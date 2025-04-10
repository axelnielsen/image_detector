"""
Script para probar el navegador automatizado.
"""

import os
import sys
import argparse
from pathlib import Path

# Añadir el directorio raíz al path para importar módulos
sys.path.append(str(Path(__file__).parent.parent))

from src.crawlers.web_crawler import WebCrawler, DarkPatternCrawler

def test_basic_navigation(url: str):
    """
    Prueba la navegación básica a una URL.
    
    Args:
        url: URL a la que navegar
    """
    print(f"\n=== Prueba de navegación básica a {url} ===")
    
    with WebCrawler(headless=True) as crawler:
        print("Navegador iniciado")
        
        # Navegar a la URL
        print(f"Navegando a {url}...")
        success = crawler.navigate(url)
        print(f"Navegación exitosa: {success}")
        
        if success:
            # Obtener título de la página
            title = crawler.page.title()
            print(f"Título de la página: {title}")
            
            # Tomar captura de pantalla
            screenshot_path = crawler.take_screenshot()
            print(f"Captura de pantalla guardada en: {screenshot_path}")
            
            # Obtener estructura DOM
            print("Obteniendo estructura DOM...")
            dom = crawler.get_dom_structure()
            print(f"Tipo de nodo raíz: {dom['type']}")
            print(f"Número de nodos hijos: {len(dom.get('children', []))}")
    
    print("Navegador cerrado")
    print("✓ Prueba de navegación básica completada\n")

def test_interaction(url: str):
    """
    Prueba interacciones con elementos de la página.
    
    Args:
        url: URL a la que navegar
    """
    print(f"\n=== Prueba de interacción con {url} ===")
    
    with WebCrawler(headless=True) as crawler:
        print("Navegador iniciado")
        
        # Navegar a la URL
        print(f"Navegando a {url}...")
        success = crawler.navigate(url)
        print(f"Navegación exitosa: {success}")
        
        if success:
            # Esperar a que la página cargue completamente
            crawler.wait_for_navigation()
            
            # Desplazarse por la página
            print("Desplazándose por la página...")
            crawler.scroll_to_bottom(step=300, delay=0.2)
            
            # Buscar un enlace para hacer clic
            print("Buscando enlaces en la página...")
            links = crawler.find_elements('a')
            print(f"Se encontraron {len(links)} enlaces")
            
            if links:
                # Tomar captura de pantalla antes de hacer clic
                before_click = crawler.take_screenshot(name="before_click")
                print(f"Captura antes de clic guardada en: {before_click}")
                
                # Hacer clic en el primer enlace visible
                for i, link in enumerate(links[:5]):  # Intentar con los primeros 5 enlaces
                    try:
                        # Verificar si el enlace es visible
                        is_visible = link.is_visible()
                        if is_visible:
                            href = link.get_attribute('href')
                            text = link.text_content().strip()
                            print(f"Haciendo clic en enlace: {text or href}")
                            link.click()
                            print("Clic exitoso")
                            
                            # Esperar a que la navegación se complete
                            crawler.wait_for_navigation()
                            
                            # Tomar captura de pantalla después de hacer clic
                            after_click = crawler.take_screenshot(name="after_click")
                            print(f"Captura después de clic guardada en: {after_click}")
                            break
                    except Exception as e:
                        print(f"No se pudo hacer clic en el enlace {i}: {e}")
    
    print("Navegador cerrado")
    print("✓ Prueba de interacción completada\n")

def test_dark_pattern_crawler(url: str):
    """
    Prueba el crawler especializado en patrones oscuros.
    
    Args:
        url: URL a la que navegar
    """
    print(f"\n=== Prueba de DarkPatternCrawler en {url} ===")
    
    with DarkPatternCrawler(headless=True) as crawler:
        print("Navegador especializado iniciado")
        
        # Analizar la página
        print(f"Analizando {url}...")
        results = crawler.analyze_page(url)
        
        if results['success']:
            print(f"Análisis exitoso de {url}")
            print(f"Título: {results['title']}")
            print(f"Capturas de pantalla:")
            for name, path in results['screenshots'].items():
                print(f"  - {name}: {path}")
            
            # Guardar evidencia de ejemplo
            evidence_path = crawler.save_evidence(
                evidence_type="test_evidence",
                data={
                    "test_key": "test_value",
                    "screenshot_paths": results['screenshots']
                },
                description="Esta es una evidencia de prueba"
            )
            print(f"Evidencia guardada en: {evidence_path}")
        else:
            print(f"Error al analizar {url}: {results.get('error')}")
    
    print("Navegador especializado cerrado")
    print("✓ Prueba de DarkPatternCrawler completada\n")

def main():
    parser = argparse.ArgumentParser(description='Prueba el navegador automatizado')
    parser.add_argument('--url', type=str, default='https://www.example.com',
                        help='URL para pruebas')
    args = parser.parse_args()
    
    try:
        print("Iniciando pruebas del navegador automatizado...")
        
        # Prueba de navegación básica
        test_basic_navigation(args.url)
        
        # Prueba de interacción
        test_interaction(args.url)
        
        # Prueba del crawler especializado
        test_dark_pattern_crawler(args.url)
        
        print("✅ Todas las pruebas completadas con éxito!")
    except Exception as e:
        print(f"❌ Error en las pruebas: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
