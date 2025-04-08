import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

class PersonaDetector:
    """
    Detector para identificar si una imagen contiene personas.
    Utiliza un modelo preentrenado SSD MobileNet V2 para detección de objetos.
    """
    
    def __init__(self, confidence_threshold=0.5):
        """
        Inicializa el detector de personas.
        
        Args:
            confidence_threshold (float): Umbral de confianza para considerar una detección válida.
        """
        self.confidence_threshold = confidence_threshold
        
        # Cargar modelo SSD MobileNet v2 desde TensorFlow Hub
        self.detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
        
        # ID de clase "person" en COCO es 1
        self.person_class_id = 1

    def detect(self, image):
        """
        Detecta si la imagen contiene personas.
        
        Args:
            image: Imagen en formato NumPy array (BGR).
            
        Returns:
            dict: Resultado de la detección:
                {
                    "present": bool,
                    "confidence": float,
                    "bounding_boxes": list  # Cada box: [ymin, xmin, ymax, xmax]
                }
        """
        # Convertir BGR a RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor([rgb_image], dtype=tf.uint8)

        # Ejecutar el modelo
        detections = self.detector(input_tensor)

        # Extraer resultados
        classes = detections['detection_classes'][0].numpy().astype(np.int32)
        scores = detections['detection_scores'][0].numpy()
        boxes = detections['detection_boxes'][0].numpy()

        person_detected = False
        max_confidence = 0.0
        person_boxes = []

        for cls, score, box in zip(classes, scores, boxes):
            if cls == self.person_class_id and score >= self.confidence_threshold:
                person_detected = True
                max_confidence = max(max_confidence, score)
                person_boxes.append(box.tolist())

        return {
            "present": person_detected,
            "confidence": float(max_confidence),
            "bounding_boxes": person_boxes
        }

    def visualize(self, image, result):
        """
        Dibuja el resultado de la detección sobre la imagen.
        
        Args:
            image: Imagen original en formato NumPy array (BGR).
            result: Resultado de la detección.
            
        Returns:
            image: Imagen con visualización del resultado.
        """
        vis_image = image.copy()
        height, width, _ = vis_image.shape

        for box in result.get("bounding_boxes", []):
            ymin, xmin, ymax, xmax = box
            top_left = (int(xmin * width), int(ymin * height))
            bottom_right = (int(xmax * width), int(ymax * height))
            color = (0, 255, 0)
            cv2.rectangle(vis_image, top_left, bottom_right, color, 2)
        
        text = f"Persona: {'Sí' if result['present'] else 'No'}"
        confidence_text = f"Confianza: {result['confidence']:.2f}"
        color = (0, 255, 0) if result['present'] else (0, 0, 255)

        cv2.putText(vis_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(vis_image, confidence_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return vis_image
