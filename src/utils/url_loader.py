"""
Módulo para cargar URLs desde diferentes formatos de archivo.
Soporta CSV, JSON y TXT.
"""

import csv
import json
import os
import re
from typing import List, Dict, Any, Optional
import pandas as pd
from urllib.parse import urlparse


class URLValidator:
    """Clase para validar URLs."""
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """
        Valida si una URL tiene un formato correcto.
        
        Args:
            url: La URL a validar
            
        Returns:
            bool: True si la URL es válida, False en caso contrario
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except:
            return False


class URLLoader:
    """Clase para cargar URLs desde diferentes formatos de archivo."""
    
    def __init__(self, file_path: str):
        """
        Inicializa el cargador de URLs.
        
        Args:
            file_path: Ruta al archivo que contiene las URLs
        """
        self.file_path = file_path
        self.validator = URLValidator()
        
    def load(self) -> List[str]:
        """
        Carga URLs desde el archivo especificado.
        
        Returns:
            List[str]: Lista de URLs válidas
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"El archivo {self.file_path} no existe")
        
        file_extension = os.path.splitext(self.file_path)[1].lower()
        
        if file_extension == '.csv':
            return self._load_from_csv()
        elif file_extension == '.json':
            return self._load_from_json()
        elif file_extension == '.txt':
            return self._load_from_txt()
        else:
            raise ValueError(f"Formato de archivo no soportado: {file_extension}")
    
    def _load_from_csv(self) -> List[str]:
        """
        Carga URLs desde un archivo CSV.
        
        Returns:
            List[str]: Lista de URLs válidas
        """
        urls = []
        try:
            # Intentar usar pandas para mayor flexibilidad
            df = pd.read_csv(self.file_path)
            
            # Buscar columna que contenga URLs
            url_column = None
            for col in df.columns:
                if 'url' in col.lower() or 'link' in col.lower() or 'site' in col.lower():
                    url_column = col
                    break
            
            if url_column:
                urls = df[url_column].tolist()
            else:
                # Si no hay columna específica, usar la primera columna
                urls = df.iloc[:, 0].tolist()
        except:
            # Fallback a CSV básico si pandas falla
            with open(self.file_path, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row and len(row) > 0:
                        urls.append(row[0])
        
        return [url for url in urls if self.validator.is_valid_url(url)]
    
    def _load_from_json(self) -> List[str]:
        """
        Carga URLs desde un archivo JSON.
        
        Returns:
            List[str]: Lista de URLs válidas
        """
        urls = []
        with open(self.file_path, 'r') as file:
            data = json.load(file)
            
            # Manejar diferentes estructuras JSON
            if isinstance(data, list):
                # Si es una lista de strings
                if all(isinstance(item, str) for item in data):
                    urls = data
                # Si es una lista de objetos
                elif all(isinstance(item, dict) for item in data):
                    for item in data:
                        for key, value in item.items():
                            if (isinstance(value, str) and 
                                ('url' in key.lower() or 'link' in key.lower() or 'site' in key.lower())):
                                urls.append(value)
            # Si es un objeto
            elif isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str):
                        if ('url' in key.lower() or 'link' in key.lower() or 'site' in key.lower()):
                            urls.append(value)
                    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                        if ('url' in key.lower() or 'link' in key.lower() or 'site' in key.lower()):
                            urls.extend(value)
        
        return [url for url in urls if self.validator.is_valid_url(url)]
    
    def _load_from_txt(self) -> List[str]:
        """
        Carga URLs desde un archivo TXT.
        
        Returns:
            List[str]: Lista de URLs válidas
        """
        urls = []
        with open(self.file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):  # Ignorar líneas vacías y comentarios
                    urls.append(line)
        
        return [url for url in urls if self.validator.is_valid_url(url)]


class URLQueue:
    """Clase para gestionar una cola de URLs a procesar."""
    
    def __init__(self, urls: List[str] = None):
        """
        Inicializa la cola de URLs.
        
        Args:
            urls: Lista inicial de URLs
        """
        self.queue = urls or []
        self.processed = []
        self.failed = []
        
    def add(self, url: str) -> None:
        """
        Añade una URL a la cola.
        
        Args:
            url: URL a añadir
        """
        if url not in self.queue and url not in self.processed and url not in self.failed:
            self.queue.append(url)
    
    def add_batch(self, urls: List[str]) -> None:
        """
        Añade un lote de URLs a la cola.
        
        Args:
            urls: Lista de URLs a añadir
        """
        for url in urls:
            self.add(url)
    
    def get_next(self) -> Optional[str]:
        """
        Obtiene la siguiente URL a procesar.
        
        Returns:
            str: La siguiente URL o None si la cola está vacía
        """
        if not self.queue:
            return None
        return self.queue[0]
    
    def mark_processed(self, url: str) -> None:
        """
        Marca una URL como procesada.
        
        Args:
            url: URL a marcar como procesada
        """
        if url in self.queue:
            self.queue.remove(url)
            self.processed.append(url)
    
    def mark_failed(self, url: str) -> None:
        """
        Marca una URL como fallida.
        
        Args:
            url: URL a marcar como fallida
        """
        if url in self.queue:
            self.queue.remove(url)
            self.failed.append(url)
    
    def is_empty(self) -> bool:
        """
        Comprueba si la cola está vacía.
        
        Returns:
            bool: True si la cola está vacía, False en caso contrario
        """
        return len(self.queue) == 0
    
    def get_stats(self) -> Dict[str, int]:
        """
        Obtiene estadísticas de la cola.
        
        Returns:
            Dict[str, int]: Diccionario con estadísticas
        """
        return {
            'pending': len(self.queue),
            'processed': len(self.processed),
            'failed': len(self.failed),
            'total': len(self.queue) + len(self.processed) + len(self.failed)
        }
