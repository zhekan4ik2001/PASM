import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import color, io

def lines_cross_point(line1, line2):
    # unpack points coordinates:
    (((x1, y1), (x2, y2)), ((x3, y3), (x4, y4))) = (line1, line2)
    (dx1, dy1) = (x1 - x2, y1 - y2)
    (dx2, dy2) = (x3 - x4, y3 - y4)
    det = dx1 * dy2 - dx2 * dy1
    if det == 0:
        return None
    det_x = (x1 * y2 - y1 * x2) * dx2 - (x3 * y4 - y3 * x4) * dx1
    det_y = (x1 * y2 - y1 * x2) * dy2 - (x3 * y4 - y3 * x4) * dy1
    return (det_x / det, det_y / det)

# определяем координаты вершин треугольника цветового охвата:
XYZ_R = color.rgb2xyz(np.concatenate(([255], [0], [0])))
(x_R, y_R) = (XYZ_R[0], XYZ_R[1]) / (XYZ_R[0] + XYZ_R[1] + XYZ_R[2])
XYZ_G = color.rgb2xyz(np.concatenate(([0], [255], [0])))
(x_G, y_G) = (XYZ_G[0], XYZ_G[1]) / (XYZ_G[0] + XYZ_G[1] + XYZ_G[2])
XYZ_B = color.rgb2xyz(np.concatenate(([0], [0], [255])))
(x_B, y_B) = (XYZ_B[0], XYZ_B[1]) / (XYZ_B[0] + XYZ_B[1] + XYZ_B[2])

image_in = cv2.imread("img.jpg")
(M, N, K) = image_in.shape
image_RGB = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)

# Преобразование изображения из RGB в XYZ
image_XYZ = color.rgb2xyz(image_RGB)
(x_WP, y_WP) = (0.333, 0.333)  # точка белого цвета

# Преобразование из XYZ в Yxy
x = np.zeros_like(image_XYZ[:, :, 0]) + x_WP
y = np.zeros_like(image_XYZ[:, :, 0]) + y_WP
denom_XYZ = image_XYZ[:, :, 0] + image_XYZ[:, :, 1] + image_XYZ[:, :, 2]
x = np.divide(image_XYZ[:, :, 0], denom_XYZ, where=denom_XYZ != 0)
y = np.divide(image_XYZ[:, :, 1], denom_XYZ, where=denom_XYZ != 0)
Y = image_XYZ[:, :, 1]

# ── Жёлтый сектор (входной — для анализа) ──
(x_yellow_left_in, y_yellow_left_in) = (0.4, 0.6)
(x_yellow_right_in, y_yellow_right_in) = (0.45, 0.5)
angle_yellow_left_in = np.angle((x_yellow_left_in - x_WP) + 1j * (y_yellow_left_in - y_WP))
angle_yellow_right_in = np.angle((x_yellow_right_in - x_WP) + 1j * (y_yellow_right_in - y_WP))
angle_yellow_sector_in = abs(angle_yellow_right_in - angle_yellow_left_in)

# ── Жёлтый сектор (выходной — для синтеза) ──
(x_yellow_left_out, y_yellow_left_out) = (0.45, 0.55)
(x_yellow_right_out, y_yellow_right_out) = (0.48, 0.48)
angle_yellow_left_out = np.angle((x_yellow_left_out - x_WP) + 1j * (y_yellow_left_out - y_WP))
angle_yellow_right_out = np.angle((x_yellow_right_out - x_WP) + 1j * (y_yellow_right_out - y_WP))
angle_yellow_sector_out = abs(angle_yellow_right_out - angle_yellow_left_out)

# ── Красный сектор (входной — для анализа) ──
# Левая граница совпадает с правой границей жёлтого, чтобы сектора не перекрывались
(x_red_left_in, y_red_left_in) = (0.50, 0.34)
(x_red_right_in, y_red_right_in) = (0.58, 0.22)
angle_red_left_in = np.angle((x_red_left_in - x_WP) + 1j * (y_red_left_in - y_WP))
angle_red_right_in = np.angle((x_red_right_in - x_WP) + 1j * (y_red_right_in - y_WP))
angle_red_sector_in = abs(angle_red_right_in - angle_red_left_in)

# ── Красный сектор (выходной — для синтеза) ──
(x_red_left_out, y_red_left_out) = (0.68, 0.41)
(x_red_right_out, y_red_right_out) = (0.72, 0.29)
angle_red_left_out = np.angle((x_red_left_out - x_WP) + 1j * (y_red_left_out - y_WP))
angle_red_right_out = np.angle((x_red_right_out - x_WP) + 1j * (y_red_right_out - y_WP))
angle_red_sector_out = abs(angle_red_right_out - angle_red_left_out)

# Угол на красный пиксель — нужен для выбора ребра гамута (G-R или B-R),
# т.к. красная вершина лежит на стыке двух рёбер:
angle_red_primary = np.angle((x_R - x_WP) + 1j * (y_R - y_WP))

# рассчитываем коэффициенты уравнения прямых типа y = kx,
# приняв точку белого цвета (WP) за начало координат:
k_yellow_left = (y_yellow_left_in - y_WP) / (x_yellow_left_in - x_WP)
k_yellow_right = (y_yellow_right_in - y_WP) / (x_yellow_right_in - x_WP)
k_red_left = (y_red_left_in - y_WP) / (x_red_left_in - x_WP)
k_red_right = (y_red_right_in - y_WP) / (x_red_right_in - x_WP)

(is_yellow, is_red) = (np.zeros_like(x), np.zeros_like(x))

# анализируем пиксели изображения, собирая статистику по заменяемым цветам:
(min_Y_yellow, max_Y_yellow) = (1, 0)
(min_Y_red, max_Y_red) = (1, 0)

for m in range(M):
    for n in range(N):
        is_yellow[m, n] = ((x[m, n] > x_WP) &
            (y[m, n] < y_WP + k_yellow_left * (x[m, n] - x_WP)) &
            (y[m, n] > y_WP + k_yellow_right * (x[m, n] - x_WP)))
        is_red[m, n] = ((x[m, n] > x_WP) &
            (y[m, n] < y_WP + k_red_left * (x[m, n] - x_WP)) &
            (y[m, n] > y_WP + k_red_right * (x[m, n] - x_WP)))
        if is_yellow[m, n]:
            if Y[m, n] > max_Y_yellow:
                max_Y_yellow = Y[m, n]
            if Y[m, n] < min_Y_yellow:
                min_Y_yellow = Y[m, n]
        if is_red[m, n]:
            if Y[m, n] > max_Y_red:
                max_Y_red = Y[m, n]
            if Y[m, n] < min_Y_red:
                min_Y_red = Y[m, n]

# альтернативный продвинутый подход:
Y_yellow_sorted = np.sort(Y[is_yellow > 0])
Y_red_sorted = np.sort(Y[is_red > 0])
if len(Y_yellow_sorted) > 0:
    max_Y_yellow = Y_yellow_sorted[min(int(0.98 * len(Y_yellow_sorted)), len(Y_yellow_sorted) - 1)]
if len(Y_red_sorted) > 0:
    max_Y_red = Y_red_sorted[min(int(0.98 * len(Y_red_sorted)), len(Y_red_sorted) - 1)]

# вычислим размахи яркостей:
delta_Y_yellow = max_Y_yellow - min_Y_yellow
delta_Y_red = max_Y_red - min_Y_red

# ── Преобразование цветностей и яркостей пикселей ──
for m in range(M):
    for n in range(N):
        if is_red[m, n]:
            # ── Красный → Жёлтый ──
            input_distance_abs = np.sqrt((x[m, n] - x_WP) ** 2 + (y[m, n] - y_WP) ** 2)
            input_angle = np.angle((x[m, n] - x_WP) + 1j * (y[m, n] - y_WP))

            if input_angle > angle_red_primary:
                (x_edge, y_edge) = lines_cross_point(
                    ((x_WP, y_WP), (x[m, n], y[m, n])),
                    ((x_G, y_G), (x_R, y_R)))
            else:
                (x_edge, y_edge) = lines_cross_point(
                    ((x_WP, y_WP), (x[m, n], y[m, n])),
                    ((x_B, y_B), (x_R, y_R)))

            if x_edge is not None:
                distance_W_edge = np.sqrt((x_edge - x_WP) ** 2 + (y_edge - y_WP) ** 2)
                input_distance_rel = input_distance_abs / distance_W_edge
            else:
                input_distance_rel = 1.0

            input_angle_abs = abs(input_angle - angle_red_left_in)
            input_angle_rel = input_angle_abs / angle_red_sector_in

            output_angle_abs = angle_yellow_left_out - input_angle_rel * angle_yellow_sector_out

            (x_temp, y_temp) = (x_WP + np.cos(output_angle_abs),
                                y_WP + np.sin(output_angle_abs))
            (x_Y, y_Y) = lines_cross_point(
                ((x_WP, y_WP), (x_temp, y_temp)),
                ((x_G, y_G), (x_R, y_R)))

            if x_Y is not None:
                distance_W_Y = np.sqrt((x_Y - x_WP) ** 2 + (y_Y - y_WP) ** 2)
            else:
                distance_W_Y = 1.0

            output_distance_abs = input_distance_rel * distance_W_Y
            x[m, n] = x_WP + output_distance_abs * np.cos(output_angle_abs)
            y[m, n] = y_WP + output_distance_abs * np.sin(output_angle_abs)

            # Усиление яркости: поднимаем Y пропорционально расстоянию от белой точки
            # Коэффициент 1.8 подобран для заметного эффекта; можно менять от 1.2 до 2.0
            scale = 1.0 + 1.8 * output_distance_abs
            Y[m, n] = np.clip(Y[m, n] * scale, 0.0, 1.0)

        if is_yellow[m, n]:
            # ── Жёлтый → Красный ──
            input_distance_abs = np.sqrt((x[m, n] - x_WP) ** 2 + (y[m, n] - y_WP) ** 2)
            (x_Y, y_Y) = lines_cross_point(
                ((x_WP, y_WP), (x[m, n], y[m, n])),
                ((x_G, y_G), (x_R, y_R)))

            if x_Y is not None:
                distance_W_Y = np.sqrt((x_Y - x_WP) ** 2 + (y_Y - y_WP) ** 2)
                input_distance_rel = input_distance_abs / distance_W_Y
            else:
                input_distance_rel = 1.0

            input_angle = np.angle((x[m, n] - x_WP) + 1j * (y[m, n] - y_WP))
            input_angle_abs = abs(input_angle - angle_yellow_left_in)
            input_angle_rel = input_angle_abs / angle_yellow_sector_in

            output_angle_abs = angle_red_left_out - input_angle_rel * angle_red_sector_out

            if output_angle_abs > angle_red_primary:
                (x_edge, y_edge) = lines_cross_point(
                    ((x_WP, y_WP), (x_WP + np.cos(output_angle_abs),
                                    y_WP + np.sin(output_angle_abs))),
                    ((x_G, y_G), (x_R, y_R)))
            else:
                (x_edge, y_edge) = lines_cross_point(
                    ((x_WP, y_WP), (x_WP + np.cos(output_angle_abs),
                                    y_WP + np.sin(output_angle_abs))),
                    ((x_B, y_B), (x_R, y_R)))

            if x_edge is not None:
                distance_W_edge = np.sqrt((x_edge - x_WP) ** 2 + (y_edge - y_WP) ** 2)
            else:
                distance_W_edge = 1.0

            output_distance_abs = input_distance_rel * distance_W_edge
            x[m, n] = x_WP + output_distance_abs * np.cos(output_angle_abs)
            y[m, n] = y_WP + output_distance_abs * np.sin(output_angle_abs)

            # Усиление яркости аналогично красному сектору
            scale = 1.0 + 1.8 * output_distance_abs
            Y[m, n] = np.clip(Y[m, n] * scale, 0.0, 1.0)



# Преобразование из Yxy обратно в XYZ
image_XYZ_new = image_XYZ.copy()
image_XYZ_new[:, :, 1] = Y
image_XYZ_new[:, :, 0] = x * (np.divide(Y, y, where=y != 0))
image_XYZ_new[:, :, 2] = (1 - x - y) * (np.divide(Y, y, where=y != 0))

# Преобразование из XYZ обратно в RGB
image_rgb_new = color.xyz2rgb(image_XYZ_new) * 255.0
image_rgb_new = image_rgb_new.astype(np.float32)
image_bgr_new = cv2.cvtColor(image_rgb_new, cv2.COLOR_RGB2BGR)
image_rgb_out = (np.clip(image_rgb_new, 0, 255)).astype(np.uint8)

# Отображение оригинального и измененного изображений
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Исходное изображение')
plt.imshow(image_RGB)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('Преобразованное изображение')
plt.imshow(image_rgb_out)
plt.axis('off')
plt.show()

# записываем полученное изображение в файл:
cv2.imwrite("image_out.jpg", image_bgr_new)
