import os
import cv2
import numpy as np
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, concatenate
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from fondo_blanco_detector import FondoBlancoDetector
from persona_detector import PersonaDetector
from cedula_detector import CedulaIdentidadDetector
from rut_detector import RutChilenoDetector

class ModelTrainer:
    """
    Clase para entrenar un modelo de detección de características en imágenes.
    Integra los detectores individuales y entrena un modelo combinado.
    """
    
    def __init__(self, dataset_dir, model_dir, img_size=(224, 224)):
        """
        Inicializa el entrenador del modelo.
        
        Args:
            dataset_dir (str): Directorio del conjunto de datos.
            model_dir (str): Directorio donde se guardarán los modelos entrenados.
            img_size (tuple): Tamaño de las imágenes para el modelo.
        """
        self.dataset_dir = Path(dataset_dir)
        self.model_dir = Path(model_dir)
        self.img_size = img_size
        
        # Crear directorio para modelos si no existe
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar detectores
        self.fondo_blanco_detector = FondoBlancoDetector()
        self.persona_detector = PersonaDetector()
        self.cedula_detector = CedulaIdentidadDetector()
        self.rut_detector = RutChilenoDetector()
        
        # Modelo combinado
        self.model = None
    
    def _load_and_preprocess_image(self, image_path):
        """
        Carga y preprocesa una imagen para el entrenamiento.
        
        Args:
            image_path (str): Ruta de la imagen.
            
        Returns:
            np.array: Imagen preprocesada.
        """
        # Cargar imagen
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # Redimensionar
        img = cv2.resize(img, self.img_size)
        
        # Normalizar
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def _extract_features(self, image):
        """
        Extrae características de una imagen usando los detectores.
        
        Args:
            image: Imagen en formato NumPy array (BGR).
            
        Returns:
            dict: Características extraídas.
        """
        # Detectar características
        fondo_result = self.fondo_blanco_detector.detect(image)
        persona_result = self.persona_detector.detect(image)
        cedula_result = self.cedula_detector.detect(image)
        rut_result = self.rut_detector.detect(image)
        
        # Crear vector de características
        features = {
            "fondo_blanco": {
                "present": fondo_result["present"],
                "confidence": fondo_result["confidence"],
                "porcentaje_blanco": fondo_result["metadata"]["porcentaje_blanco"] / 100.0
            },
            "persona": {
                "present": persona_result["present"],
                "confidence": persona_result["confidence"]
            },
            "cedula": {
                "present": cedula_result["present"],
                "confidence": cedula_result["confidence"]
            },
            "rut": {
                "present": rut_result["present"],
                "confidence": rut_result["confidence"]
            }
        }
        
        return features
    
    def _create_model(self):
        """
        Crea la arquitectura del modelo combinado.
        
        Returns:
            tf.keras.Model: Modelo combinado.
        """
        # Entrada de imagen
        input_img = Input(shape=(self.img_size[0], self.img_size[1], 3))
        
        # Usar MobileNetV2 como extractor de características
        base_model = MobileNetV2(
            input_shape=(self.img_size[0], self.img_size[1], 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Congelar capas del modelo base
        for layer in base_model.layers:
            layer.trainable = False
        
        # Características extraídas por el modelo base
        x = base_model(input_img)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        
        # Salidas para cada característica
        fondo_blanco_output = Dense(1, activation='sigmoid', name='fondo_blanco')(x)
        persona_output = Dense(1, activation='sigmoid', name='persona')(x)
        cedula_output = Dense(1, activation='sigmoid', name='cedula')(x)
        rut_output = Dense(1, activation='sigmoid', name='rut')(x)
        
        # Crear modelo con múltiples salidas
        model = Model(
            inputs=input_img,
            outputs=[fondo_blanco_output, persona_output, cedula_output, rut_output]
        )
        
        # Compilar modelo
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'fondo_blanco': 'binary_crossentropy',
                'persona': 'binary_crossentropy',
                'cedula': 'binary_crossentropy',
                'rut': 'binary_crossentropy'
            },
            metrics={
                'fondo_blanco': ['accuracy'],
                'persona': ['accuracy'],
                'cedula': ['accuracy'],
                'rut': ['accuracy']
            }
        )
        
        return model
    
    def prepare_dataset(self):
        """
        Prepara el conjunto de datos para el entrenamiento.
        
        Returns:
            tuple: (X_train, y_train, X_val, y_val)
        """
        # Listas para almacenar datos
        images = []
        labels = []
        
        # Recorrer directorios de entrenamiento
        train_dir = self.dataset_dir / 'train'
        
        for feature_dir in train_dir.iterdir():
            if not feature_dir.is_dir():
                continue
                
            feature_name = feature_dir.name
            
            # Procesar ejemplos positivos
            positive_dir = feature_dir / 'positive'
            for img_file in positive_dir.glob('*.jpg'):
                try:
                    # Cargar y preprocesar imagen
                    img = self._load_and_preprocess_image(img_file)
                    images.append(img)
                    
                    # Crear etiqueta
                    label = {
                        'fondo_blanco': 1.0 if feature_name == 'fondo_blanco' else 0.0,
                        'persona': 1.0 if feature_name == 'personas' else 0.0,
                        'cedula': 1.0 if feature_name == 'cedula_identidad' else 0.0,
                        'rut': 1.0 if feature_name == 'rut_chileno' else 0.0
                    }
                    labels.append(label)
                except Exception as e:
                    print(f"Error procesando {img_file}: {e}")
            
            # Procesar ejemplos negativos
            negative_dir = feature_dir / 'negative'
            for img_file in negative_dir.glob('*.jpg'):
                try:
                    # Cargar y preprocesar imagen
                    img = self._load_and_preprocess_image(img_file)
                    images.append(img)
                    
                    # Crear etiqueta
                    label = {
                        'fondo_blanco': 0.0,
                        'persona': 0.0,
                        'cedula': 0.0,
                        'rut': 0.0
                    }
                    labels.append(label)
                except Exception as e:
                    print(f"Error procesando {img_file}: {e}")
        
        # Convertir a arrays numpy
        X = np.array(images)
        y = {
            'fondo_blanco': np.array([label['fondo_blanco'] for label in labels]),
            'persona': np.array([label['persona'] for label in labels]),
            'cedula': np.array([label['cedula'] for label in labels]),
            'rut': np.array([label['rut'] for label in labels])
        }
        
        # Dividir en entrenamiento y validación
        indices = np.arange(len(X))
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
        
        X_train = X[train_indices]
        X_val = X[val_indices]
        
        y_train = {
            'fondo_blanco': y['fondo_blanco'][train_indices],
            'persona': y['persona'][train_indices],
            'cedula': y['cedula'][train_indices],
            'rut': y['rut'][train_indices]
        }
        
        y_val = {
            'fondo_blanco': y['fondo_blanco'][val_indices],
            'persona': y['persona'][val_indices],
            'cedula': y['cedula'][val_indices],
            'rut': y['rut'][val_indices]
        }
        
        return X_train, y_train, X_val, y_val
    
    def train(self, epochs=50, batch_size=32):
        """
        Entrena el modelo combinado.
        
        Args:
            epochs (int): Número de épocas de entrenamiento.
            batch_size (int): Tamaño del lote para entrenamiento.
            
        Returns:
            history: Historial de entrenamiento.
        """
        # Crear modelo
        self.model = self._create_model()
        
        # Preparar conjunto de datos
        X_train, y_train, X_val, y_val = self.prepare_dataset()
        
        # Callbacks para entrenamiento
        callbacks = [
            ModelCheckpoint(
                str(self.model_dir / 'best_model.h5'),
                save_best_only=True,
                monitor='val_loss'
            ),
            EarlyStopping(
                patience=10,
                monitor='val_loss'
            ),
            ReduceLROnPlateau(
                factor=0.1,
                patience=5,
                monitor='val_loss'
            )
        ]
        
        # Entrenar modelo
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        # Guardar modelo final
        self.model.save(str(self.model_dir / 'final_model.h5'))
        
        return history
    
    def evaluate(self, test_dir=None):
        """
        Evalúa el modelo en el conjunto de prueba.
        
        Args:
            test_dir (str, optional): Directorio de prueba. Si es None, usa el directorio de prueba del dataset.
            
        Returns:
            dict: Métricas de evaluación.
        """
        if test_dir is None:
            test_dir = self.dataset_dir / 'test'
        else:
            test_dir = Path(test_dir)
        
        # Cargar modelo si no está cargado
        if self.model is None:
            model_path = self.model_dir / 'best_model.h5'
            if model_path.exists():
                self.model = tf.keras.models.load_model(str(model_path))
            else:
                raise ValueError("No se encontró un modelo entrenado. Entrene primero.")
        
        # Listas para almacenar datos
        images = []
        labels = []
        
        # Recorrer directorios de prueba
        for feature_dir in test_dir.iterdir():
            if not feature_dir.is_dir():
                continue
                
            feature_name = feature_dir.name
            
            # Procesar ejemplos positivos
            positive_dir = feature_dir / 'positive'
            for img_file in positive_dir.glob('*.jpg'):
                try:
                    # Cargar y preprocesar imagen
                    img = self._load_and_preprocess_image(img_file)
                    images.append(img)
                    
                    # Crear etiqueta
                    label = {
                        'fondo_blanco': 1.0 if feature_name == 'fondo_blanco' else 0.0,
                        'persona': 1.0 if feature_name == 'personas' else 0.0,
                        'cedula': 1.0 if feature_name == 'cedula_identidad' else 0.0,
                        'rut': 1.0 if feature_name == 'rut_chileno' else 0.0
                    }
                    labels.append(label)
                except Exception as e:
                    print(f"Error procesando {img_file}: {e}")
            
            # Procesar ejemplos negativos
            negative_dir = feature_dir / 'negative'
            for img_file in negative_dir.glob('*.jpg'):
                try:
                    # Cargar y preprocesar imagen
                    img = self._load_and_preprocess_image(img_file)
                    images.append(img)
                    
                    # Crear etiqueta
                    label = {
                        'fondo_blanco': 0.0,
                        'persona': 0.0,
                        'cedula': 0.0,
                        'rut': 0.0
                    }
                    labels.append(label)
                except Exception as e:
                    print(f"Error procesando {img_file}: {e}")
        
        # Convertir a arrays numpy
        X_test = np.array(images)
        y_test = {
            'fondo_blanco': np.array([label['fondo_blanco'] for label in labels]),
            'persona': np.array([label['persona'] for label in labels]),
            'cedula': np.array([label['cedula'] for label in labels]),
            'rut': np.array([label['rut'] for label in labels])
        }
        
        # Evaluar modelo
        evaluation = self.model.evaluate(X_test, y_test)
        
        # Crear diccionario de métricas
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = evaluation[i]
        
        return metrics
    
    def predict(self, image_path):
        """
        Realiza predicciones en una imagen.
        
        Args:
            image_path (str): Ruta de la imagen.
            
        Returns:
            dict: Predicciones para cada característica.
        """
        # Cargar modelo si no está cargado
        if self.model is None:
            model_path = self.model_dir / 'best_model.h5'
            if model_path.exists():
                self.model = tf.keras.models.load_model(str(model_path))
            else:
                # Si no hay modelo entrenado, usar detectores individuales
                img = cv2.imread(str(image_path))
                if img is None:
                    raise ValueError(f"No se pudo cargar la imagen: {image_path}")
                
                features = self._extract_features(img)
                
                return {
                    'fondo_blanco': features['fondo_blanco']['present'],
                    'persona': features['persona']['present'],
                    'cedula': features['cedula']['present'],
                    'rut': features['rut']['present'],
                    'confidence': {
                        'fondo_blanco': features['fondo_blanco']['confidence'],
                        'persona': features['persona']['confidence'],
                        'cedula': features['cedula']['confidence'],
                        'rut': features['rut']['confidence']
                    }
                }
        
        # Cargar y preprocesar imagen
        img = self._load_and_preprocess_image(image_path)
        img_batch = np.expand_dims(img, axis=0)
        
        # Realizar predicción
        predictions = self.model.predict(img_batch)
        
        # Formatear resultados
        result = {
            'fondo_blanco': bool(predictions[0][0] > 0.5),
            'persona': bool(predictions[1][0] > 0.5),
            'cedula': bool(predictions[2][0] > 0.5),
            'rut': bool(predictions[3][0] > 0.5),
            'confidence': {
                'fondo_blanco': float(predictions[0][0]),
                'persona': float(predictions[1][0]),
                'cedula': float(predictions[2][0]),
                'rut': float(predictions[3][0])
            }
        }
        
        return result

if __name__ == "__main__":
    # Crear entrenador
    trainer = ModelTrainer(
        dataset_dir="/home/ubuntu/detector_imagenes/dataset",
        model_dir="/home/ubuntu/detector_imagenes/models"
    )
    
    # Ejemplo de uso (comentado para evitar ejecución accidental)
    # history = trainer.train(epochs=10, batch_size=16)
    # metrics = trainer.evaluate()
    # print(metrics)
