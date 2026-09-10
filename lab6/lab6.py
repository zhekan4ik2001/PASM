from PIL import Image, ImageDraw
from moviepy.editor import ImageClip, concatenate_videoclips
import numpy as np
import random

# === Параметры видео ===
width, height = 320, 180
fps = 30
duration_sec = 30
total_frames = fps * duration_sec

# === Параметры квадратов ===
size = 10
border = 1
bg_color = (128, 128, 128)
max_squares = 30
speed_range = 2.5
spawn_delay_frames = fps * 0.3

base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


def mix_colors(c1, c2):
    return tuple(min(255, c1[i] + c2[i]) for i in range(3))


def overlaps(ax, ay, bx, by, s):
    return ax < bx + s and bx < ax + s and ay < by + s and by < ay + s


# === Создание начальных квадратов без перекрытия ===
squares = []
for color in base_colors:
    while True:
        x = random.randint(0, width - size)
        y = random.randint(0, height - size)
        ok = not any(overlaps(x, y, sq['x'], sq['y'], size + 4) for sq in squares)
        if ok:
            break
    vx = random.uniform(-speed_range, speed_range)
    vy = random.uniform(-speed_range, speed_range)
    if abs(vx) < 0.5:
        vx = random.choice([-1, 1])
    if abs(vy) < 0.5:
        vy = random.choice([-1, 1])
    squares.append({'x': float(x), 'y': float(y), 'vx': vx, 'vy': vy, 'color': color})

colliding_pairs = set()
pending_spawns = []  # отложенные создания: (frame_to_spawn, color, x, y)
frames = []

for frame_idx in range(total_frames):
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # --- Обновление позиций и отскок от стен ---
    for sq in squares:
        sq['x'] += sq['vx']
        sq['y'] += sq['vy']
        if sq['x'] <= 0:
            sq['x'] = 0
            sq['vx'] = abs(sq['vx'])
        elif sq['x'] + size >= width:
            sq['x'] = width - size
            sq['vx'] = -abs(sq['vx'])
        if sq['y'] <= 0:
            sq['y'] = 0
            sq['vy'] = abs(sq['vy'])
        elif sq['y'] + size >= height:
            sq['y'] = height - size
            sq['vy'] = -abs(sq['vy'])

    # --- Проверка столкновений ---
    current_colliding = set()
    new_spawn = None  # за один кадр — максимум одна отложенная регистрация

    for i in range(len(squares)):
        for j in range(i + 1, len(squares)):
            sq1, sq2 = squares[i], squares[j]
            if overlaps(sq1['x'], sq1['y'], sq2['x'], sq2['y'], size):
                current_colliding.add((i, j))
                if (i, j) not in colliding_pairs and new_spawn is None:
                    # Обмен скоростей
                    sq1['vx'], sq2['vx'] = sq2['vx'], sq1['vx']
                    sq1['vy'], sq2['vy'] = sq2['vy'], sq1['vy']

                    # Раздвигаем
                    dx = sq2['x'] - sq1['x']
                    dy = sq2['y'] - sq1['y']
                    if abs(dx) >= abs(dy):
                        if dx > 0:
                            sq2['x'] = sq1['x'] + size
                        else:
                            sq2['x'] = sq1['x'] - size
                    else:
                        if dy > 0:
                            sq2['y'] = sq1['y'] + size
                        else:
                            sq2['y'] = sq1['y'] - size

                    # Запоминаем точку касания для отложенного спавна
                    if len(squares) + len(pending_spawns) < max_squares:
                        cx = (max(sq1['x'], sq2['x']) +
                              min(sq1['x'], sq2['x']) + size) / 2
                        cy = (max(sq1['y'], sq2['y']) +
                              min(sq1['y'], sq2['y']) + size) / 2
                        nx = max(0, min(width - size, int(cx - size / 2)))
                        ny = max(0, min(height - size, int(cy - size / 2)))
                        mixed = mix_colors(sq1['color'], sq2['color'])
                        spawn_frame = frame_idx + spawn_delay_frames
                        new_spawn = (spawn_frame, mixed, nx, ny)

    colliding_pairs = current_colliding
    if new_spawn is not None:
        pending_spawns.append(new_spawn)

    # --- Проверка отложенных спавнов: настало время? ---
    ready = []
    remaining = []
    for ps in pending_spawns:
        if ps[0] <= frame_idx:
            ready.append(ps)
        else:
            remaining.append(ps)
    pending_spawns = remaining

    for spawn_frame, color, sx, sy in ready:
        if len(squares) < max_squares:
            vx = random.uniform(-speed_range, speed_range)
            vy = random.uniform(-speed_range, speed_range)
            if abs(vx) < 0.5:
                vx = random.choice([-1, 1])
            if abs(vy) < 0.5:
                vy = random.choice([-1, 1])
            squares.append({
                'x': float(sx), 'y': float(sy),
                'vx': vx, 'vy': vy,
                'color': color
            })

    # --- Рисуем все квадраты с чёрной рамкой ---
    for sq in squares:
        x1, y1 = int(sq['x']), int(sq['y'])
        x2, y2 = x1 + size, y1 + size
        draw.rectangle((x1, y1, x2, y2), fill=sq['color'],
                       outline=(0, 0, 0), width=border)

    frame_np = np.array(img)
    frames.append(ImageClip(frame_np, duration=1 / fps))

# === Сборка и сохранение ===
video = concatenate_videoclips(frames, method='compose')
video.write_videofile('moving_squares.mp4', fps=fps, codec='mpeg4', bitrate='500k')
