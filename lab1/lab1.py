import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import color

# Загрузка изображения
image_in = cv2.imread("img.jpg")
if image_in is None:
    raise FileNotFoundError("Файл img.png не найден")

(M, N, K) = image_in.shape

# BGR -> RGB
image_rgb = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

# RGB -> XYZ
image_xyz = color.rgb2xyz(image_rgb)

# XYZ -> Yxy
denom_xyz = image_xyz[:, :, 0] + image_xyz[:, :, 1] + image_xyz[:, :, 2]
eps = 1e-8
denom_safe = np.maximum(denom_xyz, eps)

x = image_xyz[:, :, 0] / denom_safe
y = image_xyz[:, :, 1] / denom_safe
Y = image_xyz[:, :, 1]

# Белая точка (стандарт D65)
x_WP = 0.3127
y_WP = 0.3290

# Диапазон для зелёного в Yxy (эмпирически подобран под типичные зелёные)
# Зелёный обычно правее и выше белой точки, но не слишком высоко
x_min, x_max = 0.0, 0.35
y_min, y_max = 0.4, 0.9

# Маска: пиксели, попадающие в прямоугольник в Yxy
mask_green = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

# Создаём выходное изображение
image_rgb_new = image_rgb.copy()

# Для зелёных пикселей делаем серый, сохраняя яркость Y
# В XYZ серый — это X=Y=Z, поэтому создаём фиктивный XYZ и конвертируем в RGB
xyz_gray = np.zeros_like(image_xyz)
xyz_gray[:, :, 0] = Y
xyz_gray[:, :, 1] = Y
xyz_gray[:, :, 2] = Y

rgb_gray = color.xyz2rgb(xyz_gray)

# Применяем маску
image_rgb_new[mask_green] = rgb_gray[mask_green]

# Ограничиваем диапазон [0, 1]
image_rgb_new = np.clip(image_rgb_new, 0, 1)

# Конвертируем в BGR и uint8 для записи
image_bgr_new = (cv2.cvtColor(image_rgb_new, cv2.COLOR_RGB2BGR) * 255).astype(np.uint8)

# Отображение результата
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title("Исходное изображение")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Маска зелёного (бинарная)")
plt.imshow(mask_green, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Без зелёного (заменён на серый)")
plt.imshow(image_rgb_new)
plt.axis("off")
plt.tight_layout()
plt.show()

# Сохранение результата
cv2.imwrite("img_out.jpg", image_bgr_new)
