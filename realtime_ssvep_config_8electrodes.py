# ─── 实时SSVEP控制程序配置（8电极版本） ────────────────────────────────
# 刺激频率定义（对应上下左右四个指令）
FREQ_UP    = 7.5    # Hz - 上
FREQ_RIGHT = 8.571   # Hz - 右  
FREQ_DOWN  = 10.0   # Hz - 下
FREQ_LEFT  = 12.0   # Hz - 左

# 刺激窗口配置
STIM_SIZE   = 600         # px - 刺激方块大小 (增大)
WINDOW_SIZE = (800, 800)  # px - 窗口大小 (增大)
STIM_DURATION = 2.0       # 秒 - 每个刺激持续时间 (延长一倍)
REST_DURATION = 1.0       # 秒 - 休息时间

# 脑电数据采集配置（8个通道）
FS          = 250         # Hz - 采样率

# 8个电极通道映射
# LSL通道索引到电极名称的映射：
# 通道 0: O2
# 通道 1: O1  
# 通道 2: Oz
# 通道 3: Pz
# 通道 4: P4
# 通道 5: P3
# 通道 6: P8
# 通道 7: P7

CH_O2 = 0  # O2通道索引
CH_O1 = 1  # O1通道索引  
CH_OZ = 2  # Oz通道索引
CH_PZ = 3  # Pz通道索引
CH_P4 = 4  # P4通道索引
CH_P3 = 5  # P3通道索引
CH_P8 = 6  # P8通道索引
CH_P7 = 7  # P7通道索引

# 所有通道列表
ALL_CHANNELS = [CH_O2, CH_O1, CH_OZ, CH_PZ, CH_P4, CH_P3, CH_P8, CH_P7]

# 信号处理配置
PROCESSING_WINDOW_SEC = 2.0  # 处理窗口长度（秒）
PROCESSING_HOP_SEC    = 0.5  # 处理步长（秒）
BANDPASS_LOW  = 5.0          # Hz - 带通滤波低频
BANDPASS_HIGH = 60.0         # Hz - 带通滤波高频
NOTCH_FREQ    = 50.0         # Hz - 陷波频率
NOTCH_Q       = 30.0         # 陷波Q值

# CCA算法配置
CCA_THRESHOLD = 0.3          # CCA相关系数阈值
MAX_HARMONICS = 3            # 最大谐波数

# 控制输出配置
OUTPUT_COMMANDS = True       # 是否输出控制指令
VERBOSE_OUTPUT = True        # 是否显示详细信息
# ─────────────────────────────────────────────────────────
