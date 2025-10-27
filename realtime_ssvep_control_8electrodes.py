#!/usr/bin/env python3
"""
realtime_ssvep_control_8electrodes.py
──────────────────────────────────────────────────────────────
实时SSVEP脑电控制程序（8电极版本）

使用8个电极（O2, O1, Oz, Pz, P4, P3, P8, P7）进行SSVEP识别
理论上可以提高识别率，因为：
1. 提供更多空间信息
2. SSVEP信号在不同电极上有不同的响应强度
3. CCA算法可以利用多通道数据提高特征提取能力
"""

import sys
import time
import threading
import random
import numpy as np
import tkinter as tk
from tkinter import ttk
from pylsl import StreamInlet, resolve_byprop
import scipy.signal as signal
from sklearn.cross_decomposition import CCA
import realtime_ssvep_config_8electrodes as cfg
import warnings
import os
import platform

# 声音提示模块
try:
    import winsound
    SOUND_AVAILABLE = True
    SOUND_TYPE = 'winsound'
    print("✅ 声音系统初始化完成 (winsound)")
except ImportError:
    try:
        import pygame
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        SOUND_AVAILABLE = True
        SOUND_TYPE = 'pygame'
        print("✅ 声音系统初始化完成 (pygame)")
    except ImportError:
        SOUND_AVAILABLE = False
        SOUND_TYPE = None
        print("⚠️ 声音系统不可用")

# 抑制sklearn的警告
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')

class SoundManager:
    """声音管理器"""
    
    def __init__(self):
        self.sound_available = SOUND_AVAILABLE
        self.sound_enabled = True
        self.sound_type = SOUND_TYPE
        
        if self.sound_available:
            try:
                if self.sound_type == 'winsound':
                    import winsound
                    self.winsound = winsound
                    print("✅ winsound声音系统准备完成")
                elif self.sound_type == 'pygame':
                    import pygame
                    self.pygame = pygame
                    self._init_pygame_sounds()
                    print("✅ pygame声音文件生成完成")
            except Exception as e:
                print(f"⚠️ 声音初始化失败: {e}")
                self.sound_available = False
                self.sound_enabled = False
    
    def _init_pygame_sounds(self):
        """初始化pygame声音"""
        try:
            # 生成提示音
            self.beep_sound = self._generate_beep(800, 0.5)  # 800Hz, 0.1秒
            self.success_sound = self._generate_beep(1000, 1)  # 1000Hz, 0.2秒
            self.error_sound = self._generate_beep(400, 0.3)   # 400Hz, 0.3秒
        except Exception as e:
            print(f"⚠️ pygame声音生成失败: {e}")
            self.sound_available = False
    
    def _generate_beep(self, frequency, duration):
        """生成指定频率和时长的蜂鸣声"""
        try:
            sample_rate = 22050
            frames = int(duration * sample_rate)
            arr = np.zeros(frames)
            
            for i in range(frames):
                arr[i] = np.sin(2 * np.pi * frequency * i / sample_rate)
            
            # 添加淡入淡出效果
            fade_frames = int(0.01 * sample_rate)  # 10ms淡入淡出
            for i in range(fade_frames):
                arr[i] *= i / fade_frames
                arr[-(i+1)] *= i / fade_frames
            
            # 转换为16位整数
            arr = (arr * 32767).astype(np.int16)
            
            # 创建立体声
            stereo_arr = np.zeros((frames, 2), dtype=np.int16)
            stereo_arr[:, 0] = arr
            stereo_arr[:, 1] = arr
            
            # 创建pygame声音对象
            sound = self.pygame.sndarray.make_sound(stereo_arr)
            return sound
        except Exception as e:
            print(f"⚠️ 声音生成失败: {e}")
            return None
    
    def play_beep(self):
        """播放开始提示音"""
        if not self.sound_enabled or not self.sound_available:
            return
        
        try:
            if self.sound_type == 'winsound' and hasattr(self, 'winsound'):
                self.winsound.Beep(800, 100)  # 800Hz, 100ms
            elif self.sound_type == 'pygame' and hasattr(self, 'beep_sound'):
                self.beep_sound.play()
        except Exception as e:
            if cfg.VERBOSE_OUTPUT:
                print(f"⚠️ 播放提示音失败: {e}")
    
    def play_success(self):
        """播放成功提示音"""
        if not self.sound_enabled or not self.sound_available:
            return
        
        try:
            if self.sound_type == 'winsound' and hasattr(self, 'winsound'):
                self.winsound.Beep(1000, 200)  # 1000Hz, 200ms
            elif self.sound_type == 'pygame' and hasattr(self, 'success_sound'):
                self.success_sound.play()
        except Exception as e:
            if cfg.VERBOSE_OUTPUT:
                print(f"⚠️ 播放成功音失败: {e}")
    
    def play_error(self):
        """播放错误提示音"""
        if not self.sound_enabled or not self.sound_available:
            return
        
        try:
            if self.sound_type == 'winsound' and hasattr(self, 'winsound'):
                self.winsound.Beep(400, 300)  # 400Hz, 300ms
            elif self.sound_type == 'pygame' and hasattr(self, 'error_sound'):
                self.error_sound.play()
        except Exception as e:
            if cfg.VERBOSE_OUTPUT:
                print(f"⚠️ 播放错误音失败: {e}")
    
    def toggle_sound(self):
        """切换声音开关"""
        self.sound_enabled = not self.sound_enabled
        status = "开启" if self.sound_enabled else "关闭"
        print(f"🔊 声音提示已{status}")

class SSVEPProcessor:
    """SSVEP信号处理器 - 8电极版本"""
    
    def __init__(self):
        self.fs = cfg.FS
        self.window_len = int(cfg.PROCESSING_WINDOW_SEC * self.fs)
        self.num_channels = len(cfg.ALL_CHANNELS)  # 8个通道
        
        # 设计滤波器
        self._design_filters()
        
        # 频率映射
        self.frequencies = {
            'up': cfg.FREQ_UP,
            'right': cfg.FREQ_RIGHT, 
            'down': cfg.FREQ_DOWN,
            'left': cfg.FREQ_LEFT
        }
        
        print(f"✅ SSVEP处理器初始化完成 (8电极版本)")
        print(f"   - 采样率: {self.fs} Hz")
        print(f"   - 处理窗口: {cfg.PROCESSING_WINDOW_SEC} 秒")
        print(f"   - 通道数: {self.num_channels} 个")
        print(f"   - 刺激频率: {self.frequencies}")
    
    def _design_filters(self):
        """设计滤波器"""
        nyquist = 0.5 * self.fs
        
        # 带通滤波器 (5-60Hz)
        low = cfg.BANDPASS_LOW / nyquist
        high = cfg.BANDPASS_HIGH / nyquist
        self.bp_b, self.bp_a = signal.butter(4, [low, high], btype='band')
        
        # 陷波滤波器 (50Hz)
        notch_freq = cfg.NOTCH_FREQ / nyquist
        self.notch_b, self.notch_a = signal.iirnotch(notch_freq, cfg.NOTCH_Q)
        
        print(f"✅ 滤波器设计完成")
    
    def preprocess_signal(self, signal_data):
        """信号预处理 - 加强数据验证"""
        try:
            # 检查输入数据
            if not np.isfinite(signal_data).all():
                print("⚠️ 输入信号包含无效值，进行清理")
                signal_data = np.nan_to_num(signal_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 基线校正
            signal_data = signal_data - np.mean(signal_data)
            
            # 陷波滤波
            signal_data = signal.filtfilt(self.notch_b, self.notch_a, signal_data)
            
            # 带通滤波
            signal_data = signal.filtfilt(self.bp_b, self.bp_a, signal_data)
            
            # 再次检查结果
            if not np.isfinite(signal_data).all():
                print("⚠️ 滤波后信号包含无效值，进行清理")
                signal_data = np.nan_to_num(signal_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            return signal_data
        except Exception as e:
            print(f"⚠️ 信号预处理出错: {e}")
            return np.zeros_like(signal_data)
    
    def preprocess_multichannel(self, eeg_data):
        """预处理多通道EEG数据"""
        processed_channels = []
        for ch_idx in range(eeg_data.shape[1]):
            if ch_idx in cfg.ALL_CHANNELS:
                channel_data = eeg_data[:, ch_idx]
                processed = self.preprocess_signal(channel_data)
                processed_channels.append(processed)
        return np.column_stack(processed_channels)
    
    def create_reference_signals(self, freq, n_samples):
        """创建参考信号"""
        t = np.arange(n_samples) / self.fs
        ref_signals = []
        for h in range(1, cfg.MAX_HARMONICS + 1):
            ref_signals.extend([
                np.sin(2 * np.pi * h * freq * t),
                np.cos(2 * np.pi * h * freq * t)
            ])
        return np.column_stack(ref_signals)
    
    def compute_cca_correlation(self, eeg_data, freq):
        """计算CCA相关系数 - 多通道版本"""
        try:
            n_samples = eeg_data.shape[0]
            ref_signals = self.create_reference_signals(freq, n_samples)
            
            if eeg_data.shape[0] != ref_signals.shape[0]:
                return 0.0
            
            # 数据验证
            if not np.isfinite(eeg_data).all() or not np.isfinite(ref_signals).all():
                return 0.0
            
            # 检查数据是否全为零
            if np.all(eeg_data == 0) or np.all(ref_signals == 0):
                return 0.0
            
            # 数据标准化 - 加强数值稳定性
            eeg_mean = np.mean(eeg_data, axis=0)
            eeg_std = np.std(eeg_data, axis=0)
            eeg_std[eeg_std < 1e-10] = 1.0  # 避免除零
            eeg_norm = (eeg_data - eeg_mean) / eeg_std
            
            ref_mean = np.mean(ref_signals, axis=0)
            ref_std = np.std(ref_signals, axis=0)
            ref_std[ref_std < 1e-10] = 1.0  # 避免除零
            ref_norm = (ref_signals - ref_mean) / ref_std
            
            # 再次验证标准化后的数据
            if not np.isfinite(eeg_norm).all() or not np.isfinite(ref_norm).all():
                return 0.0
            
            # 计算CCA - 使用更安全的方法
            try:
                # 检查数据维度
                if eeg_norm.shape[0] < eeg_norm.shape[1] or ref_norm.shape[0] < ref_norm.shape[1]:
                    # 如果样本数小于特征数，使用简化的相关系数计算
                    correlation = np.corrcoef(eeg_norm.flatten(), ref_norm.flatten())[0, 1]
                    return float(np.clip(abs(correlation), 0.0, 1.0))
                
                # 使用CCA - 多通道数据
                cca = CCA(n_components=1)
                cca.fit(eeg_norm, ref_norm)
                u, v = cca.transform(eeg_norm, ref_norm)
                
                # 检查变换结果
                if not np.isfinite(u).all() or not np.isfinite(v).all():
                    return 0.0
                
                correlation = np.corrcoef(u[:, 0], v[:, 0])[0, 1]
                
                # 检查相关系数
                if not np.isfinite(correlation):
                    return 0.0
                
                return float(np.clip(abs(correlation), 0.0, 1.0))
                
            except Exception as cca_error:
                # CCA失败时使用简化的相关系数计算
                try:
                    correlation = np.corrcoef(eeg_norm.flatten(), ref_norm.flatten())[0, 1]
                    if np.isfinite(correlation):
                        return float(np.clip(abs(correlation), 0.0, 1.0))
                except:
                    pass
                return 0.0
                
        except Exception as e:
            if cfg.VERBOSE_OUTPUT:
                print(f"⚠️ CCA计算出错: {e}")
            return 0.0
    
    def classify_frequency(self, eeg_data):
        """分类SSVEP频率 - 8电极版本"""
        if eeg_data is None or eeg_data.shape[0] < self.window_len:
            return None, 0.0
        
        recent_data = eeg_data[-self.window_len:]
        if recent_data.shape[1] < len(cfg.ALL_CHANNELS):
            return None, 0.0
        
        # 检查数据有效性
        if not np.isfinite(recent_data).all():
            print("⚠️ EEG数据包含无效值")
            return None, 0.0
        
        # 提取8个通道的数据
        channel_data = []
        for ch_idx in cfg.ALL_CHANNELS:
            if ch_idx < recent_data.shape[1]:
                channel_data.append(recent_data[:, ch_idx])
            else:
                print(f"⚠️ 通道 {ch_idx} 超出范围")
                return None, 0.0
        
        # 预处理每个通道
        processed_data = []
        for ch_data in channel_data:
            processed = self.preprocess_signal(ch_data)
            processed_data.append(processed)
        
        # 合并为多通道数据
        combined_data = np.column_stack(processed_data)
        
        # 最终数据验证
        if not np.isfinite(combined_data).all():
            print("⚠️ 预处理后数据包含无效值")
            return None, 0.0
        
        # 计算每个频率的CCA相关系数
        correlations = {}
        for command, freq in self.frequencies.items():
            corr = self.compute_cca_correlation(combined_data, freq)
            correlations[command] = corr
        
        # 找到最大相关系数
        if correlations:
            max_command = max(correlations, key=correlations.get)
            max_correlation = correlations[max_command]
            
            if max_correlation > cfg.CCA_THRESHOLD:
                return max_command, max_correlation
        
        return None, 0.0

class StimulusDisplay:
    """刺激显示控制器 (tkinter版本)"""
    
    def __init__(self, sound_manager=None):
        self.sound_manager = sound_manager
        self.root = tk.Tk()
        self.root.title("SSVEP实时控制 - 8电极版本 - 按ESC退出")
        self.root.geometry(f"{cfg.WINDOW_SIZE[0]}x{cfg.WINDOW_SIZE[1]}")
        self.root.geometry(f"+400+10")
        self.root.configure(bg='black')
        
        # 绑定ESC键
        self.root.bind('<Escape>', self.on_escape)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # 创建主画布
        self.canvas = tk.Canvas(
            self.root, 
            width=cfg.WINDOW_SIZE[0], 
            height=cfg.WINDOW_SIZE[1],
            bg='black',
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 计算刺激方块位置
        self.center_x = cfg.WINDOW_SIZE[0] // 2
        self.center_y = cfg.WINDOW_SIZE[1] // 2
        self.square_size = cfg.STIM_SIZE
        
        # 创建刺激方块
        self.square = self.canvas.create_rectangle(
            self.center_x - self.square_size // 2,
            self.center_y - self.square_size // 2,
            self.center_x + self.square_size // 2,
            self.center_y + self.square_size // 2,
            fill='white',
            outline='white'
        )
        
        # 创建文本标签
        self.title_text = self.canvas.create_text(
            self.center_x, 
            self.center_y - self.square_size // 2 - 50,
            text="",
            fill='green',
            font=('Arial', 24, 'bold'),
            anchor='center'
        )
        
        self.countdown_text = self.canvas.create_text(
            self.center_x, 
            self.center_y + self.square_size // 2 + 50,
            text="",
            fill='red',
            font=('Arial', 18),
            anchor='center'
        )
        
        self.freq_text = self.canvas.create_text(
            self.center_x, 
            self.center_y + self.square_size // 2 + 100,
            text="",
            fill='blue',
            font=('Arial', 14),
            anchor='center'
        )
        
        self.run = True
        
        print(f"✅ 刺激显示初始化完成 (tkinter)")
        print(f"   - 窗口大小: {cfg.WINDOW_SIZE}")
        print(f"   - 刺激方块大小: {cfg.STIM_SIZE}x{cfg.STIM_SIZE}")
    
    def on_escape(self, event):
        """ESC键处理"""
        self.run = False
        self.root.quit()
    
    def on_close(self):
        """窗口关闭处理"""
        self.run = False
        self.root.quit()
    
    def display_stimulus(self, command, freq, duration):
        """显示指定频率的刺激"""
        print(f"🔄 显示刺激: {command.upper()} ({freq} Hz)")
        
        # 更新标题
        self.canvas.itemconfig(self.title_text, text=f"{command.upper()} ({freq} Hz)")
        self.canvas.itemconfig(self.freq_text, text=f"频率: {freq} Hz")
        
        # 显示准备画面
        self.canvas.itemconfig(self.square, fill='black')
        self.canvas.itemconfig(self.countdown_text, text="准备中...")
        self.root.update()
        
        # 等待1秒并播放提示音
        print("🔊 播放开始提示音...")
        if self.sound_manager:
            self.sound_manager.play_beep()
        
        # 等待1秒
        time.sleep(1.0)
        
        # 开始刺激
        start_time = time.time()
        
        while time.time() - start_time < duration and self.run:
            # 计算闪烁相位
            elapsed = time.time() - start_time
            phase = (elapsed * freq) % 1.0
            
            # 根据相位选择颜色
            color = 'white' if phase < 0.5 else 'black'
            self.canvas.itemconfig(self.square, fill=color)
            
            # 更新倒计时
            remaining = duration - elapsed
            self.canvas.itemconfig(self.countdown_text, text=f"剩余: {remaining:.1f}s")
            
            # 更新显示
            self.root.update()
            
            # 控制帧率
            time.sleep(1/60)  # 60 FPS
        
        return self.run
    
    def display_rest(self, duration):
        """显示休息画面"""
        print(f"⏸️ 休息 {duration:.1f} 秒")
        
        start_time = time.time()
        
        # 更新显示
        self.canvas.itemconfig(self.square, fill='black')
        self.canvas.itemconfig(self.title_text, text="REST")
        self.canvas.itemconfig(self.freq_text, text="")
        
        while time.time() - start_time < duration and self.run:
            # 更新倒计时
            remaining = duration - (time.time() - start_time)
            self.canvas.itemconfig(self.countdown_text, text=f"{remaining:.1f}s")
            
            # 更新显示
            self.root.update()
            time.sleep(1/60)
        
        return self.run
    
    def cleanup(self):
        """清理资源"""
        self.run = False
        self.root.destroy()

class EEGDataCollector:
    """脑电数据收集器 - 8电极版本"""
    
    def __init__(self):
        self.eeg_data = []
        self.timestamps = []
        self.inlet = None
        self.run = True
        self.lock = threading.Lock()
        self.data_available = False
        
        # 连接LSL流
        self._connect_lsl()
        
        if self.inlet:
            threading.Thread(target=self._collect_data, daemon=True).start()
            print(f"✅ 脑电数据收集器初始化完成 (8电极)")
        else:
            print(f"❌ 未找到LSL流，程序将退出")
            sys.exit(1)
    
    def _connect_lsl(self):
        """连接LSL流"""
        try:
            print("⏳ 寻找EEG LSL流...")
            streams = resolve_byprop('type', 'EEG', timeout=10)
            if streams:
                self.inlet = StreamInlet(streams[0], max_buflen=1)
                num_channels = streams[0].channel_count()
                print(f"✅ 已连接到: {streams[0].name()}")
                print(f"   - 采样率: {streams[0].nominal_srate()} Hz")
                print(f"   - 通道数: {num_channels}")
                
                # 检查是否有足够的通道
                if num_channels < len(cfg.ALL_CHANNELS):
                    print(f"⚠️ 警告: LSL流只有 {num_channels} 个通道，但需要 {len(cfg.ALL_CHANNELS)} 个")
                    print(f"   请确保LSL流提供8个通道")
            else:
                print("❌ 未找到EEG LSL流")
                print("   请确保OpenBCI-GUI正在运行并开启了LSL流")
        except Exception as e:
            print(f"❌ 连接LSL流时出错: {e}")
    
    def _collect_data(self):
        """收集脑电数据 - 加强数据验证"""
        while self.run:
            try:
                chunk, ts = self.inlet.pull_chunk(timeout=0.1)
                if ts and chunk:
                    # 数据验证
                    chunk_array = np.array(chunk)
                    if np.isfinite(chunk_array).all():
                        with self.lock:
                            self.eeg_data.extend(chunk)
                            self.timestamps.extend(ts)
                            self.data_available = True
                            
                            # 限制缓冲区大小
                            max_samples = int(10 * cfg.FS)  # 保留10秒的数据
                            if len(self.eeg_data) > max_samples:
                                self.eeg_data = self.eeg_data[-max_samples:]
                                self.timestamps = self.timestamps[-max_samples:]
                    else:
                        print("⚠️ 检测到无效的EEG数据，已跳过")
            except Exception as e:
                if cfg.VERBOSE_OUTPUT:
                    print(f"⚠️ 收集数据时出错: {e}")
                time.sleep(0.1)
    
    def get_recent_data(self, duration_sec):
        """获取最近指定时长的数据"""
        with self.lock:
            if not self.eeg_data:
                return None
            
            n_samples = int(duration_sec * cfg.FS)
            if len(self.eeg_data) >= n_samples:
                return np.array(self.eeg_data[-n_samples:])
            else:
                return np.array(self.eeg_data)
    
    def wait_for_data(self, min_samples):
        """等待足够的数据"""
        print(f"⏳ 等待脑电数据...")
        start_time = time.time()
        while len(self.eeg_data) < min_samples and self.run:
            if time.time() - start_time > 30:  # 30秒超时
                print("❌ 等待数据超时")
                return False
            time.sleep(0.1)
        
        if len(self.eeg_data) >= min_samples:
            print(f"✅ 已收集到 {len(self.eeg_data)} 个样本")
            return True
        return False

class RealtimeSSVEPController:
    """实时SSVEP控制器主类 - 8电极版本"""
    
    def __init__(self):
        self.sound_manager = SoundManager()
        self.processor = SSVEPProcessor()
        self.display = StimulusDisplay(self.sound_manager)
        self.data_collector = EEGDataCollector()
        self.run = True
        
        self.command_map = {
            'up': 'UP',
            'right': 'RIGHT', 
            'down': 'DOWN',
            'left': 'LEFT'
        }
        
        print(f"✅ 实时SSVEP控制器初始化完成 (8电极版本)")
    
    def process_command(self, command):
        """处理识别出的控制指令"""
        if cfg.OUTPUT_COMMANDS:
            print(f"🎯 控制指令: {self.command_map.get(command, 'UNKNOWN')}")
    
    def run_session(self, num_trials=10):
        """运行SSVEP控制会话"""
        print(f"\n🚀 开始SSVEP控制会话 (共 {num_trials} 个试验) - 8电极版本")
        print("=" * 50)
        
        # 等待足够的数据
        min_samples = int(cfg.PROCESSING_WINDOW_SEC * cfg.FS)
        if not self.data_collector.wait_for_data(min_samples):
            print("❌ 无法获取足够的脑电数据")
            return
        
        commands = list(self.processor.frequencies.keys())
        correct_count = 0
        
        for trial in range(num_trials):
            if not self.run:
                break
            
            print(f"\n📋 试验 {trial + 1}/{num_trials}")
            
            # 随机选择指令
            command = random.choice(commands)
            freq = self.processor.frequencies[command]
            
            # 显示刺激
            if not self.display.display_stimulus(command, freq, cfg.STIM_DURATION):
                break
            
            # 处理脑电数据
            eeg_data = self.data_collector.get_recent_data(cfg.PROCESSING_WINDOW_SEC)
            
            if eeg_data is not None:
                predicted_command, confidence = self.processor.classify_frequency(eeg_data)
                
                if predicted_command:
                    print(f"✅ 识别结果: {predicted_command.upper()} (置信度: {confidence:.3f})")
                    
                    if predicted_command == command:
                        print(f"🎉 识别正确!")
                        # 播放成功提示音
                        if self.sound_manager:
                            self.sound_manager.play_success()
                        correct_count += 1
                        self.process_command(predicted_command)
                    else:
                        print(f"❌ 识别错误 (实际: {command.upper()})")
                        # 播放错误提示音
                        if self.sound_manager:
                            self.sound_manager.play_error()
                else:
                    print(f"❓ 未能识别出有效指令")
                    # 播放错误提示音
                    if self.sound_manager:
                        self.sound_manager.play_error()
            else:
                print(f"⚠️ 没有足够的脑电数据进行分析")
                # 播放错误提示音
                if self.sound_manager:
                    self.sound_manager.play_error()
            
            # 休息
            if trial < num_trials - 1:
                if not self.display.display_rest(cfg.REST_DURATION):
                    break
        
        # 显示准确率
        accuracy = correct_count / num_trials * 100
        print(f"\n📊 识别准确率: {correct_count}/{num_trials} ({accuracy:.1f}%)")
        print(f"🏁 SSVEP控制会话结束")
    
    def cleanup(self):
        """清理资源"""
        self.run = False
        self.display.cleanup()
        print("✅ 资源清理完成")

def main():
    """主函数"""
    print("🎯 实时SSVEP脑电控制程序 (8电极版本)")
    print("=" * 50)
    print("功能: 使用8个电极（O2, O1, Oz, Pz, P4, P3, P8, P7）进行SSVEP识别")
    print("刺激频率: 7.5Hz(上) 8.51Hz(右) 10Hz(下) 12Hz(左)")
    print("声音提示: 开始前1秒提示音，识别正确/错误不同音效")
    print("按ESC键退出程序")
    print("=" * 50)
    
    controller = None
    try:
        controller = RealtimeSSVEPController()
        controller.run_session(num_trials=8)
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断程序")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if controller:
            controller.cleanup()
        print("👋 程序结束")

if __name__ == "__main__":
    main()

