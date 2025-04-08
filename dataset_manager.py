import os
import json
import shutil
from pathlib import Path

class DatasetManager:
    """
    Clase para gestionar el conjunto de datos para el sistema de detección de imágenes.
    Permite crear la estructura de directorios, generar archivos de metadatos y gestionar
    las imágenes y etiquetas.
    """
    
    def __init__(self, base_dir):
        """
        Inicializa el gestor de conjunto de datos.
        
        Args:
            base_dir (str): Directorio base donde se almacenará el conjunto de datos.
        """
        self.base_dir = Path(base_dir)
        self.train_dir = self.base_dir / 'train'
        self.test_dir = self.base_dir / 'test'
        self.features = ['fondo_blanco', 'personas', 'cedula_identidad', 'rut_chileno']
        self.classes = ['positive', 'negative']
        
    def create_directory_structure(self):
        """
        Crea la estructura de directorios para el conjunto de datos.
        """
        # Crear directorios para cada característica y clase
        for feature in self.features:
            for split in ['train', 'test']:
                for cls in self.classes:
                    dir_path = self.base_dir / split / feature / cls
                    dir_path.mkdir(parents=True, exist_ok=True)
                    
        print(f"Estructura de directorios creada en {self.base_dir}")
    
    def generate_empty_manifests(self):
        """
        Genera archivos de manifiesto vacíos para entrenamiento y prueba.
        """
        train_manifest = {
            "dataset_name": "detector_imagenes_train",
            "num_samples": 0,
            "features": self.features,
            "samples": []
        }
        
        test_manifest = {
            "dataset_name": "detector_imagenes_test",
            "num_samples": 0,
            "features": self.features,
            "samples": []
        }
        
        # Guardar manifiestos
        with open(self.base_dir / 'train_manifest.json', 'w', encoding='utf-8') as f:
            json.dump(train_manifest, f, indent=2, ensure_ascii=False)
            
        with open(self.base_dir / 'test_manifest.json', 'w', encoding='utf-8') as f:
            json.dump(test_manifest, f, indent=2, ensure_ascii=False)
            
        # Crear archivo de estadísticas
        stats = {
            "total_images": 0,
            "train_images": 0,
            "test_images": 0,
            "features": {
                feature: {
                    "train": {"positive": 0, "negative": 0},
                    "test": {"positive": 0, "negative": 0}
                } for feature in self.features
            }
        }
        
        with open(self.base_dir / 'dataset_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
            
        print("Archivos de manifiesto y estadísticas generados")
    
    def create_sample_label(self, filename, feature, is_positive, split='train'):
        """
        Crea una etiqueta de muestra para una imagen.
        
        Args:
            filename (str): Nombre del archivo de imagen.
            feature (str): Característica principal de la imagen.
            is_positive (bool): Si es un ejemplo positivo o negativo.
            split (str): División del conjunto de datos ('train' o 'test').
            
        Returns:
            dict: Etiqueta en formato JSON.
        """
        # Determinar la ruta relativa
        cls = "positive" if is_positive else "negative"
        rel_path = f"{split}/{feature}/{cls}/{filename}"
        
        # Crear estructura básica de etiqueta
        label = {
            "filename": filename,
            "path": str(self.base_dir / rel_path),
            "features": {
                f: {
                    "present": (f == feature and is_positive),
                    "confidence": 0.95 if (f == feature and is_positive) else 0.05
                } for f in self.features
            },
            "annotations": {
                "bounding_boxes": []
            }
        }
        
        # Añadir metadatos específicos según la característica
        if feature == "fondo_blanco" and is_positive:
            label["features"]["fondo_blanco"]["metadata"] = {
                "porcentaje_blanco": 90.5
            }
        elif feature == "personas" and is_positive:
            label["annotations"]["bounding_boxes"].append({
                "label": "persona",
                "coordinates": [100, 50, 400, 500]  # [x1, y1, x2, y2]
            })
        elif feature == "cedula_identidad" and is_positive:
            label["annotations"]["bounding_boxes"].append({
                "label": "cedula",
                "coordinates": [50, 100, 350, 250]
            })
        elif feature == "rut_chileno" and is_positive:
            label["annotations"]["bounding_boxes"].append({
                "label": "rut",
                "coordinates": [120, 180, 280, 210]
            })
            
        return label
    
    def generate_sample_labels(self):
        """
        Genera etiquetas de muestra para cada característica.
        """
        for feature in self.features:
            for is_positive in [True, False]:
                for split in ['train', 'test']:
                    cls = "positive" if is_positive else "negative"
                    filename = f"sample_{feature}_{cls}.jpg"
                    
                    # Crear etiqueta
                    label = self.create_sample_label(filename, feature, is_positive, split)
                    
                    # Guardar etiqueta individual
                    label_path = self.base_dir / split / feature / cls / f"{filename.replace('.jpg', '.json')}"
                    with open(label_path, 'w', encoding='utf-8') as f:
                        json.dump(label, f, indent=2, ensure_ascii=False)
        
        print("Etiquetas de muestra generadas")

if __name__ == "__main__":
    # Crear gestor de conjunto de datos
    dataset_manager = DatasetManager("/home/ubuntu/detector_imagenes/dataset")
    
    # Crear estructura de directorios
    dataset_manager.create_directory_structure()
    
    # Generar archivos de manifiesto
    dataset_manager.generate_empty_manifests()
    
    # Generar etiquetas de muestra
    dataset_manager.generate_sample_labels()
    
    print("Estructura del conjunto de datos creada exitosamente")
