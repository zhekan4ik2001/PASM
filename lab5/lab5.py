import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import os

input_file = 'input.mp4'
output_file = 'video_canny.mp4'

if not os.path.isfile(input_file):
    raise FileNotFoundError(f"Файл {input_file} не найден")

clip = VideoFileClip(input_file)

try:
    duration = clip.duration

    # Параметры границ Канни
    low_threshold = 50
    high_threshold = 150

    def canny_edges(frame):
        """Применяет алгоритм Канни к одному кадру."""
        # Кадр приходит в формате RGB (moviepy)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        # Преобразуем обратно в RGB, чтобы moviepy мог сохранить
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return edges_rgb

    # Применяем к каждому кадру видео
    canny_clip = clip.fl_image(canny_edges)

    # Сохраняем результат
    canny_clip.write_videofile(
        output_file,
        codec='mpeg4',
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True
    )

    # Кадры для визуальной проверки
    times = [duration * 0.25, duration * 0.5, duration * 0.75]
    if duration < 3:
        times = [duration * 0.25, duration * 0.5, duration - 0.01]

    def show_frames(source_clip, processed_clip, times, titles=("Исходник", "Границы Канни")):
        plt.figure(figsize=(14, 5))
        for i, t in enumerate(times):
            row = i + 1
            # Оригинальный кадр
            plt.subplot(2, 3, row)
            frame = source_clip.get_frame(t) / 255.0
            plt.imshow(frame)
            plt.title(f"{titles[0]} ({t:.1f}s)")
            plt.axis('off')
            # Кадр после Канни
            plt.subplot(2, 3, row + 3)
            frame_edges = processed_clip.get_frame(t)
            plt.imshow(frame_edges, cmap='gray')
            plt.title(f"{titles[1]} ({t:.1f}s)")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    show_frames(clip, canny_clip, times)

finally:
    clip.close()
    canny_clip.close()
