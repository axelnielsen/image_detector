import cv2
import numpy as np
import os
from pathlib import Path

def create_white_background_image(output_path, size=(640, 480), white_percentage=0.9):
    """
    Crea una imagen con fondo blanco y algunos elementos de color.
    
    Args:
        output_path: Ruta donde guardar la imagen
        size: Tamaño de la imagen (ancho, alto)
        white_percentage: Porcentaje aproximado de píxeles blancos
    """
    # Crear imagen blanca
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    
    # Calcular área para elementos no blancos
    non_white_area = int((1 - white_percentage) * size[0] * size[1])
    
    # Añadir algunos rectángulos de colores
    num_rectangles = 3
    for i in range(num_rectangles):
        # Color aleatorio
        color = np.random.randint(0, 200, 3).tolist()
        
        # Tamaño y posición aleatorios
        width = np.random.randint(50, 150)
        height = np.random.randint(50, 150)
        x = np.random.randint(0, size[0] - width)
        y = np.random.randint(0, size[1] - height)
        
        # Dibujar rectángulo
        cv2.rectangle(img, (x, y), (x + width, y + height), color, -1)
    
    # Guardar imagen
    cv2.imwrite(output_path, img)
    print(f"Imagen con fondo blanco creada: {output_path}")

def create_non_white_background_image(output_path, size=(640, 480)):
    """
    Crea una imagen con fondo de color (no blanco).
    
    Args:
        output_path: Ruta donde guardar la imagen
        size: Tamaño de la imagen (ancho, alto)
    """
    # Crear imagen con fondo de color
    background_color = np.random.randint(100, 200, 3).tolist()
    img = np.ones((size[1], size[0], 3), dtype=np.uint8)
    img[:, :, 0] = background_color[0]
    img[:, :, 1] = background_color[1]
    img[:, :, 2] = background_color[2]
    
    # Añadir algunos elementos
    num_elements = 5
    for i in range(num_elements):
        # Color aleatorio
        color = np.random.randint(0, 255, 3).tolist()
        
        # Forma aleatoria (círculo o rectángulo)
        if np.random.rand() > 0.5:
            # Círculo
            center = (np.random.randint(50, size[0] - 50), np.random.randint(50, size[1] - 50))
            radius = np.random.randint(20, 50)
            cv2.circle(img, center, radius, color, -1)
        else:
            # Rectángulo
            width = np.random.randint(40, 100)
            height = np.random.randint(40, 100)
            x = np.random.randint(0, size[0] - width)
            y = np.random.randint(0, size[1] - height)
            cv2.rectangle(img, (x, y), (x + width, y + height), color, -1)
    
    # Guardar imagen
    cv2.imwrite(output_path, img)
    print(f"Imagen con fondo no blanco creada: {output_path}")

def create_cedula_image(output_path, size=(640, 480)):
    """
    Crea una imagen simulando una cédula de identidad.
    
    Args:
        output_path: Ruta donde guardar la imagen
        size: Tamaño de la imagen (ancho, alto)
    """
    # Crear imagen con fondo blanco
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    
    # Dimensiones de la cédula (proporción aproximada 1.6:1)
    cedula_width = int(size[0] * 0.8)
    cedula_height = int(cedula_width / 1.6)
    
    # Posición centrada
    x = (size[0] - cedula_width) // 2
    y = (size[1] - cedula_height) // 2
    
    # Dibujar cédula (rectángulo con borde)
    cv2.rectangle(img, (x, y), (x + cedula_width, y + cedula_height), (200, 200, 200), -1)
    cv2.rectangle(img, (x, y), (x + cedula_width, y + cedula_height), (0, 0, 0), 2)
    
    # Añadir elementos típicos de una cédula
    
    # Título
    cv2.putText(img, "REPUBLICA DE CHILE", (x + 20, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "CEDULA DE IDENTIDAD", (x + 20, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Foto
    photo_x = x + 20
    photo_y = y + 80
    photo_width = cedula_height - 100
    photo_height = photo_width
    cv2.rectangle(img, (photo_x, photo_y), (photo_x + photo_width, photo_y + photo_height), (180, 180, 180), -1)
    
    # Información personal
    info_x = photo_x + photo_width + 20
    cv2.putText(img, "Apellidos:", (info_x, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Nombres:", (info_x, y + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Nacionalidad:", (info_x, y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # RUT
    rut_y = y + cedula_height - 40
    cv2.putText(img, "RUT: 12.345.678-9", (x + 20, rut_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Guardar imagen
    cv2.imwrite(output_path, img)
    print(f"Imagen de cédula creada: {output_path}")

def create_rut_image(output_path, size=(640, 480)):
    """
    Crea una imagen con un RUT chileno visible.
    
    Args:
        output_path: Ruta donde guardar la imagen
        size: Tamaño de la imagen (ancho, alto)
    """
    # Crear imagen con fondo blanco
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    
    # Generar un RUT aleatorio válido
    rut_body = np.random.randint(1000000, 30000000)
    
    # Calcular dígito verificador
    multipliers = [2, 3, 4, 5, 6, 7]
    reversed_digits = [int(d) for d in str(rut_body)][::-1]
    
    total = 0
    for i, digit in enumerate(reversed_digits):
        total += digit * multipliers[i % len(multipliers)]
    
    remainder = total % 11
    dv = 11 - remainder
    
    if dv == 11:
        dv = '0'
    elif dv == 10:
        dv = 'K'
    else:
        dv = str(dv)
    
    # Formatear RUT
    rut = f"{rut_body:,}".replace(',', '.') + '-' + dv
    
    # Añadir texto de RUT
    cv2.putText(img, "RUT:", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    cv2.putText(img, rut, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # Añadir algunos elementos decorativos
    cv2.line(img, (40, 220), (600, 220), (0, 0, 0), 2)
    
    # Guardar imagen
    cv2.imwrite(output_path, img)
    print(f"Imagen con RUT creada: {output_path}")

def create_person_image(output_path, size=(640, 480)):
    """
    Crea una imagen simulando una persona (silueta).
    
    Args:
        output_path: Ruta donde guardar la imagen
        size: Tamaño de la imagen (ancho, alto)
    """
    # Crear imagen con fondo claro
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 240
    
    # Dibujar silueta de persona
    # Cabeza
    head_center = (size[0] // 2, size[1] // 4)
    head_radius = size[1] // 8
    cv2.circle(img, head_center, head_radius, (70, 70, 70), -1)
    
    # Cuerpo
    body_top = (head_center[0], head_center[1] + head_radius)
    body_width = head_radius * 3
    body_height = size[1] // 2
    body_left = body_top[0] - body_width // 2
    body_right = body_top[0] + body_width // 2
    body_bottom = body_top[1] + body_height
    
    # Torso
    cv2.rectangle(img, (body_left, body_top[1]), (body_right, body_bottom), (50, 50, 50), -1)
    
    # Brazos
    arm_width = body_width // 4
    cv2.rectangle(img, (body_left - arm_width, body_top[1]), (body_left, body_top[1] + body_height * 2 // 3), (60, 60, 60), -1)
    cv2.rectangle(img, (body_right, body_top[1]), (body_right + arm_width, body_top[1] + body_height * 2 // 3), (60, 60, 60), -1)
    
    # Piernas
    leg_width = body_width // 3
    leg_left_x = body_top[0] - leg_width
    leg_right_x = body_top[0] + leg_width - leg_width // 2
    cv2.rectangle(img, (leg_left_x, body_bottom), (leg_left_x + leg_width // 2, body_bottom + body_height // 2), (40, 40, 40), -1)
    cv2.rectangle(img, (leg_right_x, body_bottom), (leg_right_x + leg_width // 2, body_bottom + body_height // 2), (40, 40, 40), -1)
    
    # Guardar imagen
    cv2.imwrite(output_path, img)
    print(f"Imagen de persona creada: {output_path}")

def create_test_images():
    """
    Crea un conjunto de imágenes de prueba para evaluar el sistema.
    """
    base_dir = Path("/home/ubuntu/detector_imagenes/test_images")
    base_dir.mkdir(exist_ok=True)
    
    # Crear imágenes con fondo blanco
    create_white_background_image(str(base_dir / "fondo_blanco_1.jpg"))
    create_white_background_image(str(base_dir / "fondo_blanco_2.jpg"), white_percentage=0.95)
    
    # Crear imágenes sin fondo blanco
    create_non_white_background_image(str(base_dir / "no_fondo_blanco_1.jpg"))
    create_non_white_background_image(str(base_dir / "no_fondo_blanco_2.jpg"))
    
    # Crear imágenes de cédula
    create_cedula_image(str(base_dir / "cedula_1.jpg"))
    create_cedula_image(str(base_dir / "cedula_2.jpg"), size=(800, 600))
    
    # Crear imágenes con RUT
    create_rut_image(str(base_dir / "rut_1.jpg"))
    create_rut_image(str(base_dir / "rut_2.jpg"), size=(800, 600))
    
    # Crear imágenes de personas
    create_person_image(str(base_dir / "persona_1.jpg"))
    create_person_image(str(base_dir / "persona_2.jpg"), size=(800, 600))
    
    # Crear imágenes combinadas
    # Persona con cédula
    img = np.ones((600, 800, 3), dtype=np.uint8) * 240
    # Dibujar persona (silueta simplificada)
    cv2.circle(img, (200, 150), 50, (70, 70, 70), -1)  # Cabeza
    cv2.rectangle(img, (150, 200), (250, 400), (50, 50, 50), -1)  # Cuerpo
    # Dibujar cédula
    cv2.rectangle(img, (400, 150), (700, 350), (200, 200, 200), -1)
    cv2.rectangle(img, (400, 150), (700, 350), (0, 0, 0), 2)
    cv2.putText(img, "CEDULA DE IDENTIDAD", (420, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "RUT: 12.345.678-9", (420, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imwrite(str(base_dir / "persona_con_cedula.jpg"), img)
    
    print("Imágenes de prueba creadas exitosamente.")

if __name__ == "__main__":
    create_test_images()
