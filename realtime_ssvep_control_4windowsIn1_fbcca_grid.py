#!/usr/bin/env python3
"""
realtime_ssvep_control_4windowsIn1_fbcca_grid.py

Based on `realtime_ssvep_control_4windowsIn1_fbcca.py`.
Changes:
 - The central area is now a 5x5 grid.
 - A cartoon image `res\\xiaohui.png` is placed in the center cell and moves one cell
   in the direction of recognized commands (up/right/down/left).

Usage: run in the project root where `res/xiaohui.png` and LSL stream are available.
"""

import time
import os
import threading
import pygame
import numpy as np

from realtime_ssvep_control_8electrodes_fbcca import SSVEPProcessor, EEGDataCollector, SoundManager
import realtime_ssvep_config_8electrodes as cfg

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


class FourWindowGridController:
    """Four flicker windows + central 5x5 grid with movable image"""

    def __init__(self, window_title="SSVEP 4-window FBCCA (Grid)", fps=60):
        pygame.init()
        self.fps = fps
        self.clock = pygame.time.Clock()
        self.title = window_title

        info = pygame.display.Info()
        self.screen_w = info.current_w
        self.screen_h = info.current_h

        # windowed mode sized to screen for portability
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption(self.title)

        # colors
        self.bg_color = (0, 0, 0)
        self.on_color = (255, 255, 255)
        self.off_color = (0, 0, 0)
        self.text_color = (0, 255, 0)

        # stimulus box sizing
        self.box_size = int(min(self.screen_w, self.screen_h) * 0.18)
        self.box_distance = int(min(self.screen_w, self.screen_h) * 0.4)

        cx = self.screen_w // 2
        cy = self.screen_h // 2
        s = self.box_size

        # targets placement
        self.targets = {
            'up':    pygame.Rect(cx - s//2, cy - s//2 - self.box_distance, s, s),
            'right': pygame.Rect(cx + self.box_distance - s//2, cy - s//2, s, s),
            'down':  pygame.Rect(cx - s//2, cy + self.box_distance - s//2, s, s),
            'left':  pygame.Rect(cx - self.box_distance - s//2, cy - s//2, s, s)
        }

        self.freq_map = {
            'up': cfg.FREQ_UP,
            'right': cfg.FREQ_RIGHT,
            'down': cfg.FREQ_DOWN,
            'left': cfg.FREQ_LEFT
        }

        # central grid settings
        self.grid_rows = 5
        self.grid_cols = 5
        self.grid_size = int(min(self.screen_w, self.screen_h) * 0.35)  # central square
        self.grid_left = cx - self.grid_size // 2
        self.grid_top = cy - self.grid_size // 2
        self.cell_w = self.grid_size // self.grid_cols
        self.cell_h = self.grid_size // self.grid_rows

        # image position in grid (row, col) centered
        self.img_row = self.grid_rows // 2
        self.img_col = self.grid_cols // 2

        # load image
        self.img_surf = None
        self._load_image()

        # recognition state
        self.predicted_command = None
        self.pred_confidence = 0.0
        self.last_rec_time = 0.0
        self.last_move_time = 0.0

        # processor and collector
        self.sound = SoundManager()
        self.processor = SSVEPProcessor()
        self.collector = EEGDataCollector()

        self.running = True
        self.rec_interval = cfg.PROCESSING_WINDOW_SEC
        threading.Thread(target=self._recognition_loop, daemon=True).start()

        print(f"✅ FourWindowGridController initialized: screen {self.screen_w}x{self.screen_h}")

    def _load_image(self):
        path = os.path.join('res', 'xiaohui.png')
        if not os.path.exists(path):
            print(f"⚠️ 图片未找到: {path}")
            self.img_surf = None
            return

        try:
            if PIL_AVAILABLE:
                img = Image.open(path).convert('RGBA')
                # scale to fit cell
                target_w = max(8, self.cell_w - 8)
                target_h = max(8, self.cell_h - 8)
                img = img.resize((target_w, target_h), Image.LANCZOS)
                mode = img.mode
                data = img.tobytes()
                self.img_surf = pygame.image.fromstring(data, img.size, mode)
            else:
                self.img_surf = pygame.image.load(path).convert_alpha()
                self.img_surf = pygame.transform.smoothscale(self.img_surf, (max(8, self.cell_w - 8), max(8, self.cell_h - 8)))
        except Exception as e:
            print(f"⚠️ 加载图片失败: {e}")
            self.img_surf = None

    def _recognition_loop(self):
        while self.running:
            eeg = self.collector.get_recent_data(cfg.PROCESSING_WINDOW_SEC)
            if eeg is None or eeg.size == 0:
                time.sleep(0.1)
                continue

            pred, conf = self.processor.classify_frequency(eeg)
            if pred:
                self.predicted_command = pred
                self.pred_confidence = conf
                self.last_rec_time = time.time()
                self.collector.push_marker(f'recognition_result_{pred}_{conf:.3f}')
                if cfg.OUTPUT_COMMANDS:
                    print(f"🔔 Detected: {pred.upper()} (confidence {conf:.3f})")
            else:
                self.predicted_command = None
                self.pred_confidence = 0.0
                self.collector.push_marker('recognition_unknown')

            time.sleep(self.rec_interval)

    def _maybe_move_image(self):
        # Move the image once per new recognition (use last_rec_time)
        if self.predicted_command and self.last_rec_time > self.last_move_time:
            cmd = self.predicted_command
            if cmd == 'up':
                self.img_row = max(0, self.img_row - 1)
            elif cmd == 'down':
                self.img_row = min(self.grid_rows - 1, self.img_row + 1)
            elif cmd == 'left':
                self.img_col = max(0, self.img_col - 1)
            elif cmd == 'right':
                self.img_col = min(self.grid_cols - 1, self.img_col + 1)

            self.last_move_time = self.last_rec_time

    def draw(self):
        now = time.time()
        self.screen.fill(self.bg_color)

        # draw flicker targets
        for name, rect in self.targets.items():
            freq = self.freq_map.get(name, 7.5)
            phase = (now * freq) % 1.0
            color = self.on_color if phase < 0.5 else self.off_color
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (80, 80, 80), rect, width=2)

        # draw central grid
        # background rect
        grid_rect = pygame.Rect(self.grid_left, self.grid_top, self.grid_size, self.grid_size)
        pygame.draw.rect(self.screen, (30, 30, 30), grid_rect)
        pygame.draw.rect(self.screen, (180, 180, 0), grid_rect, width=3)

        # draw cells
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                cell_rect = pygame.Rect(
                    self.grid_left + c * self.cell_w,
                    self.grid_top + r * self.cell_h,
                    self.cell_w,
                    self.cell_h
                )
                pygame.draw.rect(self.screen, (60, 60, 60), cell_rect, width=1)

        # maybe move image if new recognition
        self._maybe_move_image()

        # draw image
        if self.img_surf:
            img_x = self.grid_left + self.img_col * self.cell_w + (self.cell_w - self.img_surf.get_width()) // 2
            img_y = self.grid_top + self.img_row * self.cell_h + (self.cell_h - self.img_surf.get_height()) // 2
            self.screen.blit(self.img_surf, (img_x, img_y))

        # also show last recognized command text below grid
        font = pygame.font.SysFont('Arial', 28)
        if self.predicted_command:
            text = f"{self.predicted_command.upper()} ({self.pred_confidence:.2f})"
        else:
            text = "-- NOT DETECTED --"
        txt_surf = font.render(text, True, self.text_color)
        txt_rect = txt_surf.get_rect(center=(self.screen_w // 2, self.grid_top + self.grid_size + 30))
        self.screen.blit(txt_surf, txt_rect)

        # small instruction
        small_font = pygame.font.SysFont('Arial', 20)
        info_surf = small_font.render('Press ESC to quit', True, (200, 200, 200))
        self.screen.blit(info_surf, (10, 10))

        pygame.display.flip()

    def run(self):
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
        self.running = False
        # save data
        if getattr(cfg, 'SAVE_EEG_ON_EXIT', True) and self.collector:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            data_dir = 'data'
            os.makedirs(data_dir, exist_ok=True)
            filename = os.path.join(data_dir, f'calib_data_fbcca_4win_grid_{timestamp}.npz')
            try:
                self.collector.save_data(filename)
            except Exception:
                pass

        try:
            pygame.quit()
        except Exception:
            pass


def main():
    ctrl = FourWindowGridController()
    try:
        ctrl.run()
    except KeyboardInterrupt:
        print('User interrupted')
    finally:
        ctrl.cleanup()


if __name__ == '__main__':
    main()
