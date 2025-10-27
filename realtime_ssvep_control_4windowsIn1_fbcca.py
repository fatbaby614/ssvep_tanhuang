#!/usr/bin/env python3
"""
realtime_ssvep_control_4windows_fbcca.py
--------------------------------------
Four-window real-time SSVEP controller (FBCCA)

Description:
- Four small regions flash at the four screen directions (up/right/down/left).
    Frequencies: 7.5Hz (up), 8.51Hz (right), 10Hz (down), 12Hz (left).
- Reuses the FBCCA processor and EEG data collector from
    `realtime_ssvep_control_8electrodes_fbcca.py`.
- Displays the real-time recognized command (UP/RIGHT/DOWN/LEFT) on screen.

Run: Execute this script in an environment with an LSL EEG stream.
Dependencies: pygame, pylsl, numpy, scipy, scikit-learn, etc.
"""

import time
import os
import sys
import threading
import pygame
import numpy as np

# 导入已有模块中的处理器和数据采集器
from realtime_ssvep_control_8electrodes_fbcca import SSVEPProcessor, EEGDataCollector, SoundManager
import realtime_ssvep_config_8electrodes as cfg


class FourWindowStimController:
    """使用 pygame 在屏幕上绘制四个小窗口同时闪烁，并显示实时识别结果。"""

    def __init__(self, window_title="SSVEP 4-window FBCCA", fps=60):
        pygame.init()
        self.fps = fps
        self.clock = pygame.time.Clock()
        self.title = window_title

        # 尝试使用全屏或可配置窗口
        info = pygame.display.Info()
        self.screen_w = info.current_w
        self.screen_h = info.current_h

        # 创建窗口（windowed mode sized to screen for portability）
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption(self.title)

        # 颜色
        self.bg_color = (0, 0, 0)
        self.on_color = (255, 255, 255)
        self.off_color = (0, 0, 0)
        self.text_color = (0, 255, 0)

        # 刺激方块尺寸（相对屏幕）
        self.box_size = int(min(self.screen_w, self.screen_h) * 0.18)
        self.box_margin = int(self.box_size * 0.3)
        # 与中心的距离（用于把四个窗口分开，避免重叠）
        self.box_distance = int(min(self.screen_w, self.screen_h) * 0.4)

        # 四个刺激位置: 上、右、下、左 相对于屏幕中心
        cx = self.screen_w // 2
        cy = self.screen_h // 2
        s = self.box_size

        # 将四个刺激放在以屏幕中心为基准的四个方向，使用 box_distance 增加间距
        self.targets = {
            'up':    pygame.Rect(cx - s//2, cy - s//2 - self.box_distance, s, s),
            'right': pygame.Rect(cx + self.box_distance - s//2, cy - s//2, s, s),
            'down':  pygame.Rect(cx - s//2, cy + self.box_distance - s//2, s, s),
            'left':  pygame.Rect(cx - self.box_distance - s//2, cy - s//2, s, s)
        }

        # 频率映射（与配置一致）
        self.freq_map = {
            'up': cfg.FREQ_UP,
            'right': cfg.FREQ_RIGHT,
            'down': cfg.FREQ_DOWN,
            'left': cfg.FREQ_LEFT
        }

        # 实时识别文本
        self.predicted_command = None
        self.pred_confidence = 0.0
        self.last_rec_time = 0.0

        # 处理器与数据采集器
        self.sound = SoundManager()
        self.processor = SSVEPProcessor()
        self.collector = EEGDataCollector()

        # 控制运行
        self.running = True

        # 识别间隔：每处理窗口长度做一次判定
        self.rec_interval = cfg.PROCESSING_WINDOW_SEC
        self._start_recognition_thread()

        print(f"✅ Four-window stim controller initialized: screen {self.screen_w}x{self.screen_h}, box {self.box_size}px")

    def _start_recognition_thread(self):
        t = threading.Thread(target=self._recognition_loop, daemon=True)
        t.start()

    def _recognition_loop(self):
        """后台识别线程：周期性读取最近窗口数据并调用FBCCA分类器。"""
        while self.running:
            # 等待直到有足够数据
            eeg_data = self.collector.get_recent_data(cfg.PROCESSING_WINDOW_SEC)
            if eeg_data is None or eeg_data.size == 0:
                time.sleep(0.1)
                continue

            pred, conf = self.processor.classify_frequency(eeg_data)
            if pred:
                # 记录识别结果与标记
                self.predicted_command = pred
                self.pred_confidence = conf
                self.last_rec_time = time.time()
                self.collector.push_marker(f'recognition_result_{pred}_{conf:.3f}')
                if cfg.OUTPUT_COMMANDS:
                    print(f"🔔 Detected: {pred.upper()} (confidence {conf:.3f})")
            else:
                # 未识别
                self.predicted_command = None
                self.pred_confidence = 0.0
                self.collector.push_marker('recognition_unknown')

            # 等待下一个识别周期
            time.sleep(self.rec_interval)

    def draw(self):
        now = time.time()
        self.screen.fill(self.bg_color)

        # 绘制每个目标：根据频率计算相位
        for name, rect in self.targets.items():
            freq = self.freq_map.get(name, 7.5)
            phase = (now * freq) % 1.0
            color = self.on_color if phase < 0.5 else self.off_color
            pygame.draw.rect(self.screen, color, rect)

            # 如果当前识别与该target一致，绘制绿色边框
            if self.predicted_command == name:
                pygame.draw.rect(self.screen, (0, 255, 0), rect, width=6)
            else:
                pygame.draw.rect(self.screen, (80, 80, 80), rect, width=2)

        # draw recognition text (centered)
        font = pygame.font.SysFont('Arial', 72)
        small_font = pygame.font.SysFont('Arial', 28)

        if self.predicted_command:
            text = f"{self.predicted_command.upper()}  ({self.pred_confidence:.2f})"
        else:
            text = "-- NOT DETECTED --"

        text_surf = font.render(text, True, self.text_color)
        text_rect = text_surf.get_rect(center=(self.screen_w // 2, self.screen_h // 2))

        # draw semi-transparent background for readability
        try:
            # create a slightly larger rect for background
            bg_rect = text_rect.inflate(40, 20)
            s = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            s.fill((0, 0, 0, 160))  # semi-transparent black
            self.screen.blit(s, bg_rect.topleft)
        except Exception:
            # fallback: no background
            pass

        self.screen.blit(text_surf, text_rect)

        # show small instruction text
        info = "Press ESC to quit"
        info_surf = small_font.render(info, True, (200, 200, 200))
        self.screen.blit(info_surf, (10, 10))

        pygame.display.flip()

    def run(self):
        # 播放开始提示音
        if self.sound:
            self.sound.play_beep()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False

            self.draw()
            self.clock.tick(self.fps)

        self.cleanup()

    def cleanup(self):
        # 停止线程循环
        self.running = False

        # 保存数据（如果配置允许）
        if getattr(cfg, 'SAVE_EEG_ON_EXIT', True) and self.collector:
            # 生成文件名
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            data_dir = 'data'
            os.makedirs(data_dir, exist_ok=True)
            filename = os.path.join(data_dir, f'calib_data_fbcca_4win_{timestamp}.npz')
            try:
                self.collector.save_data(filename)
            except Exception:
                pass

        try:
            pygame.quit()
        except Exception:
            pass


def main():
    print("Starting: Four-window SSVEP real-time control (FBCCA) — four small windows flashing")
    controller = FourWindowStimController()
    try:
        controller.run()
    except KeyboardInterrupt:
        print("User interrupted")
    finally:
        controller.cleanup()
        print("Exit")


if __name__ == '__main__':
    main()
