import cv2
import numpy as np

class CedulaIdentidadDetector:
    def __init__(self, min_area_ratio=0.05, max_area_ratio=0.95, aspect_ratio_range=(1.3, 1.8)):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.aspect_ratio_range = aspect_ratio_range

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours_data = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        is_cedula = False
        confidence = 0.0
        bounding_box = None
        image_area = image.shape[0] * image.shape[1]

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                area_ratio = (w * h) / image_area
                aspect_ratio = w / h

                if (self.min_area_ratio <= area_ratio <= self.max_area_ratio and
                    self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):

                    ideal_aspect = sum(self.aspect_ratio_range) / 2
                    aspect_confidence = 1.0 - min(abs(aspect_ratio - ideal_aspect) / 0.5, 1.0)

                    ideal_area = 0.5
                    area_confidence = 1.0 - min(abs(area_ratio - ideal_area) / 0.4, 1.0)

                    current_confidence = (aspect_confidence + area_confidence) / 2

                    if current_confidence > confidence:
                        is_cedula = True
                        confidence = current_confidence
                        bounding_box = [x, y, x + w, y + h]

        result = {
            "present": is_cedula,
            "confidence": float(confidence)
        }

        if bounding_box:
            result["bounding_box"] = bounding_box

        return result

    def visualize(self, image, result):
        vis_image = image.copy()
        text = f"Cédula: {'Sí' if result['present'] else 'No'}"
        confidence_text = f"Confianza: {result['confidence']:.2f}"
        color = (0, 255, 0) if result['present'] else (0, 0, 255)

        cv2.putText(vis_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(vis_image, confidence_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if result["present"] and "bounding_box" in result:
            x1, y1, x2, y2 = result["bounding_box"]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        return vis_image
