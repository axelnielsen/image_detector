"""
Script para probar el sistema de carga de URLs.
"""

import os
import sys
import argparse
from pathlib import Path

# Añadir el directorio raíz al path para importar módulos
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.url_loader import URLLoader, URLQueue, URLValidator

def test_url_validator():
    """Prueba la validación de URLs."""
    validator = URLValidator()
    
    # URLs válidas
    valid_urls = [
        "https://www.example.com",
        "http://example.com",
        "https://subdomain.example.co.uk/path?query=value"
    ]
    
    # URLs inválidas
    invalid_urls = [
        "example.com",  # Sin protocolo
        "ftp://example.com",  # Protocolo no soportado
        "https://",  # Sin dominio
        "not a url at all"
    ]
    
    print("=== Prueba de validación de URLs ===")
    for url in valid_urls:
        result = validator.is_valid_url(url)
        print(f"URL: {url} - Válida: {result}")
        assert result is True, f"La URL {url} debería ser válida"
    
    for url in invalid_urls:
        result = validator.is_valid_url(url)
        print(f"URL: {url} - Válida: {result}")
        assert result is False, f"La URL {url} debería ser inválida"
    
    print("✓ Todas las pruebas de validación pasaron correctamente\n")

def test_url_loader(data_dir):
    """
    Prueba la carga de URLs desde diferentes formatos de archivo.
    
    Args:
        data_dir: Directorio que contiene los archivos de prueba
    """
    formats = ['csv', 'json', 'txt']
    
    print("=== Prueba de carga de URLs ===")
    for fmt in formats:
        file_path = os.path.join(data_dir, f"sample_urls.{fmt}")
        
        if not os.path.exists(file_path):
            print(f"Advertencia: El archivo {file_path} no existe, omitiendo prueba")
            continue
        
        print(f"\nCargando URLs desde {file_path}:")
        loader = URLLoader(file_path)
        try:
            urls = loader.load()
            print(f"Se cargaron {len(urls)} URLs válidas:")
            for url in urls:
                print(f"  - {url}")
            assert len(urls) > 0, f"No se cargaron URLs desde {file_path}"
        except Exception as e:
            print(f"Error al cargar URLs desde {file_path}: {e}")
            raise
    
    print("\n✓ Todas las pruebas de carga pasaron correctamente\n")

def test_url_queue():
    """Prueba la gestión de la cola de URLs."""
    print("=== Prueba de cola de URLs ===")
    
    # Inicializar con algunas URLs
    initial_urls = [
        "https://www.example1.com",
        "https://www.example2.com",
        "https://www.example3.com"
    ]
    
    queue = URLQueue(initial_urls)
    
    # Verificar estado inicial
    stats = queue.get_stats()
    print(f"Estado inicial: {stats}")
    assert stats['pending'] == 3, "Debería haber 3 URLs pendientes"
    assert stats['processed'] == 0, "No debería haber URLs procesadas"
    
    # Procesar una URL
    url = queue.get_next()
    print(f"Siguiente URL a procesar: {url}")
    queue.mark_processed(url)
    
    # Verificar después de procesar
    stats = queue.get_stats()
    print(f"Después de procesar una URL: {stats}")
    assert stats['pending'] == 2, "Debería haber 2 URLs pendientes"
    assert stats['processed'] == 1, "Debería haber 1 URL procesada"
    
    # Marcar una URL como fallida
    url = queue.get_next()
    print(f"Siguiente URL a procesar: {url}")
    queue.mark_failed(url)
    
    # Verificar después de fallar
    stats = queue.get_stats()
    print(f"Después de marcar una URL como fallida: {stats}")
    assert stats['pending'] == 1, "Debería haber 1 URL pendiente"
    assert stats['failed'] == 1, "Debería haber 1 URL fallida"
    
    # Añadir más URLs
    queue.add("https://www.example4.com")
    queue.add_batch(["https://www.example5.com", "https://www.example6.com"])
    
    # Verificar después de añadir
    stats = queue.get_stats()
    print(f"Después de añadir más URLs: {stats}")
    assert stats['pending'] == 4, "Debería haber 4 URLs pendientes"
    
    print("✓ Todas las pruebas de cola pasaron correctamente\n")

def main():
    parser = argparse.ArgumentParser(description='Prueba el sistema de carga de URLs')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directorio que contiene los archivos de prueba')
    args = parser.parse_args()
    
    # Convertir a ruta absoluta si es relativa
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', args.data_dir))
    
    print(f"Usando directorio de datos: {args.data_dir}\n")
    
    try:
        test_url_validator()
        test_url_loader(args.data_dir)
        test_url_queue()
        print("✅ Todas las pruebas completadas con éxito!")
    except Exception as e:
        print(f"❌ Error en las pruebas: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
