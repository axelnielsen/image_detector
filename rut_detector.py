import cv2
import re
import easyocr

class RutChilenoDetector:
    def __init__(self, confidence_threshold=0.6):
        self.confidence_threshold = confidence_threshold
        self.rut_patterns = [
            r'\d{1,2}\.\d{3}\.\d{3}-[\dkK]',
            r'\d{7,8}-[\dkK]',
        ]
        self.reader = easyocr.Reader(['es'], gpu=False)

    def _validate_rut(self, rut):
        rut = rut.replace('.', '').replace('-', '')
        if len(rut) <= 1:
            return False
        body, dv = rut[:-1], rut[-1].upper()
        try:
            body = int(body)
        except ValueError:
            return False
        if dv == 'K':
            dv_value = 10
        else:
            try:
                dv_value = int(dv)
            except ValueError:
                return False
        multipliers = [2, 3, 4, 5, 6, 7]
        reversed_digits = [int(d) for d in str(body)][::-1]
        total = sum(d * multipliers[i % len(multipliers)] for i, d in enumerate(reversed_digits))
        calculated_dv = 11 - (total % 11)
        if calculated_dv == 11: calculated_dv = 0
        if calculated_dv == 10: calculated_dv = 10
        return calculated_dv == dv_value

    def detect(self, image):
        results = self.reader.readtext(image)
    
        rut_found = None
        confidence = 0.0
        bounding_box = None
    
        for bbox, text, conf in results:
            raw_text = text.strip().replace('\n', '').replace('\r', '')
            clean_text = raw_text.replace(' ', '').replace('RUN', '').replace('RUT', '').replace(':', '')
            
            # Imprimir para depuración
            print(f"OCR detectó: '{raw_text}' -> Limpio: '{clean_text}' (conf: {conf:.2f})")
            
            for pattern in self.rut_patterns:
                match = re.search(pattern, clean_text)
                if match:
                    rut_candidate = match.group(0)
                    rut_found=rut_candidate
                    break
            if rut_found:
                break
    
        result = {
            "present": rut_found is not None,
            "confidence": float(confidence)
        }
    
        if rut_found:
            result["rut"] = rut_found
        if bounding_box:
            result["bounding_box"] = bounding_box    
        return result



    def visualize(self, image, result):
        vis_image = image.copy()
        text = f"RUN Chileno: {'Sí' if result['present'] else 'No'}"
        confidence_text = f"Confianza: {result['confidence']:.2f}"
        color = (0, 255, 0) if result['present'] else (0, 0, 255)

        cv2.putText(vis_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(vis_image, confidence_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if result.get("run"):
            rut_text = f"RUT: {result['run']}"
            cv2.putText(vis_image, rut_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if result.get("bounding_box"):
            x1, y1, x2, y2 = result["bounding_box"]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        return vis_image
