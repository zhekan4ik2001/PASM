import numpy as np
import cv2
import matplotlib.pyplot as plt

def blur_green_regions(image_path, output_path=None, blur_kernel=(31, 31)):
    """
    Загружает изображение, находит зелёные области в HSV и применяет к ним
    сильное размытие. Остальные части остаются без изменений.
    """
    image_in = cv2.imread(image_path)
    if image_in is None:
        raise FileNotFoundError(f"Файл {image_path} не найден")

    image_rgb = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image_in, cv2.COLOR_BGR2HSV)

    # Диапазон зелёного цвета в HSV
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([75, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)

    # Сильное размытие всего изображения
    blurred_image = cv2.GaussianBlur(image_in, blur_kernel, 0)

    # Комбинируем: зелёное — из размытого, остальное — из оригинала
    result_bgr = cv2.bitwise_and(blurred_image, blurred_image, mask=mask)
    original_non_green = cv2.bitwise_and(image_in, image_in, mask=mask_inv)
    result_bgr = cv2.add(result_bgr, original_non_green)

    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Оригинал")
    plt.imshow(image_rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Результат)")
    plt.imshow(result_rgb)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    if output_path:
        cv2.imwrite(output_path, result_bgr)
        print(f"Результат сохранён в {output_path}")

    return result_rgb

# Пример вызова:
blur_green_regions("img.jpg", "result.jpg", blur_kernel=(45, 45))
