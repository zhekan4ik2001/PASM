import cv2
import numpy as np
import matplotlib.pyplot as plt

# ================= НАСТРОЙКИ =================
THRESHOLD_L = 0.6  # Порог обнаружения: подбирайте вручную (0.4–0.8 обычно)
# Чем выше порог — тем меньше ложных срабатываний, но можно пропустить слабые объекты.
# =============================================

# Чтение входного изображения
input_image = cv2.imread('img.png')
if input_image is None:
    raise FileNotFoundError("Не удалось загрузить img.png")
M, N, Ki = input_image.shape

if Ki > 1:
    i_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
else:
    i_image = input_image.copy()

# Чтение эталонного изображения
etalon_image = cv2.imread('etalon.png')
if etalon_image is None:
    raise FileNotFoundError("Не удалось загрузить etalon.png")
Me, Ne, Ke = etalon_image.shape

if Ke > 1:
    e_image = cv2.cvtColor(etalon_image, cv2.COLOR_BGR2GRAY)
else:
    e_image = etalon_image.copy()

# Преобразование в float32 для вычислений
x = i_image.astype(np.float32) / 255.0
b = e_image.astype(np.float32) / 255.0

# Центрирование эталона (вычитаем среднее)
b_mean = np.mean(b)
b_tilda = b - b_mean
B_norm = np.linalg.norm(b_tilda)  # Норма эталона (вместо std*sqrt(Me*Ne))
if B_norm == 0:
    raise ValueError("Эталонное изображение не содержит вариаций яркости (все пиксели равны).")

# Локальное среднее для входного изображения (окно Me x Ne)
averagePQ = np.ones((Me, Ne), dtype=np.float32) / (Me * Ne)
x_local_mean = cv2.filter2D(x, -1, averagePQ, borderType=cv2.BORDER_REFLECT_101)
x_tilda = x - x_local_mean  # Центрированное входное изображение

# Числитель: корреляция центрированных фрагментов
ro_numerator = cv2.filter2D(x_tilda, -1, b_tilda, borderType=cv2.BORDER_REFLECT_101)

# Знаменатель: локальная норма (СКО) входного изображения в окне Me x Ne
x_tilda_sq = np.square(x_tilda)
x_var_sum = cv2.filter2D(x_tilda_sq, -1, averagePQ, borderType=cv2.BORDER_REFLECT_101) * (Me * Ne)
std_x = np.sqrt(np.maximum(x_var_sum, 0))  # Защита от отрицательных значений из-за округлений

# Избегаем деления на ноль
denom = B_norm * std_x
mask = denom > 1e-8
ro_map = np.zeros_like(ro_numerator)
ro_map[mask] = ro_numerator[mask] / denom[mask]

# Ограничиваем диапазон [-1, 1] для численной стабильности
ro_map = np.clip(ro_map, -1.0, 1.0)

min_val, max_val = np.min(ro_map), np.max(ro_map)
print(f"Диапазон отклика: [{min_val:.4f}, {max_val:.4f}], порог L = {THRESHOLD_L}")

# Поиск всех пиков выше порога с подавлением немаксимумов
def non_max_suppression(score_map, template_h, template_w, threshold):
    """
    Возвращает список (y, x) координат центров обнаруженных объектов.
    Подавление немаксимумов выполняется в окне размером с шаблон.
    """
    candidates = []
    h, w = score_map.shape
    # Проходим по всем пикселям
    for y in range(h):
        for x in range(w):
            if score_map[y, x] < threshold:
                continue
            # Определяем окно поиска локального максимума вокруг (y,x)
            y_min = max(0, y - template_h // 2)
            y_max = min(h, y + template_h // 2 + 1)
            x_min = max(0, x - template_w // 2)
            x_max = min(w, x + template_w // 2 + 1)
            window = score_map[y_min:y_max, x_min:x_max]
            local_max = np.max(window)
            # Если текущий пиксель — локальный максимум в своём окне, считаем его обнаружением
            if np.abs(score_map[y, x] - local_max) < 1e-6:
                candidates.append((y, x))
    return candidates

detections = non_max_suppression(ro_map, Me, Ne, THRESHOLD_L)
num_detections = len(detections)
print(f"Найдено объектов: {num_detections}")

# Подготовка изображений для отрисовки
input_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Отрисовка кругов на исходном изображении
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].imshow(input_rgb)
axs[0].set_title(f'Исходное изображение: найдено объектов = {num_detections}')
axs[0].axis('off')

# Отрисовка на карте откликов (в цветовой схеме jet)
im_jet = axs[1].imshow(ro_map, cmap='jet')
axs[1].set_title('Карта откликов (нормированная корреляция)')
axs[1].axis('off')
plt.colorbar(im_jet, ax=axs[1], fraction=0.046, pad=0.04)

radius = max(Me, Ne) // 2  # Радиус круга примерно равен половине размера эталона

for (cy, cx) in detections:
    # Рисуем на исходном изображении
    circle_input = plt.Circle((cx, cy), radius, color='red', fill=False, linewidth=2)
    axs[0].add_artist(circle_input)

    # Рисуем на карте откликов
    circle_jet = plt.Circle((cx, cy), radius, color='white', fill=False, linewidth=1.5)
    axs[1].add_artist(circle_jet)

plt.tight_layout()
plt.show()
