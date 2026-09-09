import numpy as np
import cv2
import matplotlib.pyplot as plt

# Загрузка изображения
image_in = cv2.imread("img.jpg")
if image_in is None:
    raise FileNotFoundError("Файл img.jpg не найден")

# Преобразование BGR → RGB для корректной работы с цветами
image_rgb = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)

# Конвертация в HSV для удобной сегментации цветов
image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

# Диапазоны для красного и жёлтого в HSV
# Красный в OpenCV «разрывается» по Hue: охватывает конец и начало шкалы (170–180 и 0–10)
lower_red1 = np.array([170, 50, 50])
upper_red1 = np.array([180, 255, 255])
lower_red2 = np.array([0, 50, 50])
upper_red2 = np.array([10, 255, 255])

lower_yellow = np.array([20, 50, 50])
upper_yellow = np.array([40, 255, 255])

# Создаём маски
mask_red1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(image_hsv, lower_red2, upper_red2)
mask_red = mask_red1 | mask_red2

mask_yellow = cv2.inRange(image_hsv, lower_yellow, upper_yellow)

# Копируем HSV-изображение для изменений
image_hsv_new = image_hsv.copy()

# Меняем красный и жёлтый местами: меняем их Hue, сохраняя S и V
# Для красного ставим жёлтый Hue (середина диапазона жёлтого — 30)
image_hsv_new[mask_red > 0, 0] = 30
# Для жёлтого ставим красный Hue (усреднённо: 170 для первой части и 10 для второй — возьмём 10 как пример для «тёплого» красного)
image_hsv_new[mask_yellow > 0, 0] = 10

# Возвращаем обратно в RGB
image_rgb_new = cv2.cvtColor(image_hsv_new, cv2.COLOR_HSV2RGB)

# Отображение результатов
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Исходное изображение')
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Красный и жёлтый поменяны местами')
plt.imshow(image_rgb_new)
plt.axis('off')
plt.show()

# Сохраняем результат (в BGR для совместимости с OpenCV)
image_bgr_new = cv2.cvtColor(image_rgb_new, cv2.COLOR_RGB2BGR)
cv2.imwrite("image_out.jpg", image_bgr_new)
