import numpy as np
from pylsl import StreamInlet, resolve_streams, local_clock
import time
import sys
import os
import csv
import glob

# 配置参数
SAMPLE_DURATION = 5  # 收集5秒数据用于验证
FS_CONFIG = 250  # 配置文件中指定的采样率 (Hz)

print("采样率验证程序")
print(f"配置文件中设置的采样率: {FS_CONFIG}Hz")
print(f"将收集{SAMPLE_DURATION}秒数据来验证实际采样率")

# 连接到LSL流
def connect_to_lsl_stream():
    """
    连接到LSL EEG流
    """
    print("\n正在寻找LSL EEG流...")
    streams = resolve_streams()  # 修复：移除多余的参数
    
    # 筛选出类型为EEG的流
    eeg_streams = [stream for stream in streams if stream.type() == "EEG"]
    
    if not eeg_streams:
        print("未找到LSL EEG流。请确保OpenBCI-GUI已启动并开启了LSL功能。")
        sys.exit(1)
    
    inlet = StreamInlet(eeg_streams[0])
    print(f"已连接到LSL流: {eeg_streams[0].name()}")
    return inlet

# 计算采样率
def calculate_sampling_rate(inlet):
    """
    收集数据并计算实际采样率
    """
    print(f"\n开始收集{SAMPLE_DURATION}秒数据...")
    
    timestamps = []
    samples_collected = 0
    start_time = local_clock()
    current_time = start_time
    
    # 收集指定时长的数据
    while current_time - start_time < SAMPLE_DURATION:
        chunk, ts = inlet.pull_chunk(timeout=0.1)
        if ts and chunk:
            timestamps.extend(ts)
            samples_collected += len(ts)
        current_time = local_clock()
    
    if len(timestamps) < 2:
        print("收集的数据点不足，无法计算采样率。")
        return None, None, None
    
    # 计算实际采样率
    time_diff = timestamps[-1] - timestamps[0]
    actual_fs = (len(timestamps) - 1) / time_diff
    
    # 计算相邻采样点之间的时间差
    time_intervals = np.diff(timestamps)
    avg_interval = np.mean(time_intervals)
    std_interval = np.std(time_intervals)
    
    # 计算理论上应该收集的样本数
    theoretical_samples = int(SAMPLE_DURATION * FS_CONFIG)
    
    return actual_fs, time_intervals, theoretical_samples, samples_collected

# 从CSV文件估算采样率
def estimate_sampling_rate_from_csv():
    """
    从之前记录的CSV文件估算采样率
    """
    print("\n尝试从CSV文件估算采样率...")
    
    # 查找最近的CSV文件
    csv_dir = "eeg_recording_data"
    if not os.path.exists(csv_dir):
        print(f"未找到目录: {csv_dir}")
        # 尝试在当前目录查找CSV文件作为备选
        csv_files = glob.glob("*.csv")
        if not csv_files:
            print("当前目录也未找到CSV文件")
            return None, None
        print("在当前目录找到CSV文件，尝试使用它们估算采样率...")
        return process_csv_files(csv_files)
    
    def process_csv_files(files):
        """处理找到的CSV文件并估算采样率"""
        # 按修改时间排序，选择最新的文件
        files.sort(key=os.path.getmtime, reverse=True)
        latest_csv = files[0]
        
        print(f"使用最新的CSV文件: {os.path.basename(latest_csv)}")
        
        # 读取时间戳数据
        timestamps = []
        try:
            with open(latest_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ts = float(row['Timestamp'])
                        timestamps.append(ts)
                    except (ValueError, KeyError):
                        continue
        except Exception as e:
            print(f"读取CSV文件时出错: {e}")
            return None, None
        
        if len(timestamps) < 2:
            print("CSV文件中的时间戳数据不足，无法估算采样率")
            return None, None
        
        # 计算估算的采样率
        time_diff = timestamps[-1] - timestamps[0]
        estimated_fs = (len(timestamps) - 1) / time_diff
        
        # 计算相邻采样点之间的时间差
        time_intervals = np.diff(timestamps)
        avg_interval = np.mean(time_intervals)
        std_interval = np.std(time_intervals)
        
        return estimated_fs, time_intervals
    
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        print("未找到CSV文件")
        return None, None
    
    return process_csv_files(csv_files)
    
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        print("未找到CSV文件")
        return None, None
    
    # 按修改时间排序，选择最新的文件
    csv_files.sort(key=os.path.getmtime, reverse=True)
    latest_csv = csv_files[0]
    
    print(f"使用最新的CSV文件: {os.path.basename(latest_csv)}")
    
    # 读取时间戳数据
    timestamps = []
    try:
        with open(latest_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = float(row['Timestamp'])
                    timestamps.append(ts)
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return None, None
    
    if len(timestamps) < 2:
        print("CSV文件中的时间戳数据不足，无法估算采样率")
        return None, None
    
    # 计算估算的采样率
    time_diff = timestamps[-1] - timestamps[0]
    estimated_fs = (len(timestamps) - 1) / time_diff
    
    # 计算相邻采样点之间的时间差
    time_intervals = np.diff(timestamps)
    avg_interval = np.mean(time_intervals)
    std_interval = np.std(time_intervals)
    
    return estimated_fs, time_intervals

# 主函数
def main():
    print("采样率验证程序")
    print(f"配置文件中设置的采样率: {FS_CONFIG}Hz")
    
    try:
        # 优先尝试直接从LSL流验证采样率
        try:
            # 连接到LSL流
            print("\n=== 方法1: 直接从LSL流验证 ===")
            inlet = connect_to_lsl_stream()
            
            # 收集数据并计算采样率
            actual_fs, time_intervals, theoretical_samples, samples_collected = calculate_sampling_rate(inlet)
            
            if actual_fs:
                # 显示结果
                print("\n===== 采样率验证结果 =====")
                print(f"配置文件中设置的采样率: {FS_CONFIG}Hz")
                print(f"实际采集的采样率: {actual_fs:.2f}Hz")
                print(f"期望收集的样本数: {theoretical_samples}")
                print(f"实际收集的样本数: {samples_collected}")
                print(f"平均采样间隔: {1000 * np.mean(time_intervals):.2f}ms")
                print(f"采样间隔标准差: {1000 * np.std(time_intervals):.2f}ms")
                
                # 评估采样率误差
                error_percentage = abs(actual_fs - FS_CONFIG) / FS_CONFIG * 100
                print(f"与配置值的误差: {error_percentage:.2f}%")
                
                # 判断采样率是否符合要求
                if error_percentage < 1.0:
                    print("✅ 采样率符合要求 (误差 < 1%)")
                else:
                    print("⚠️ 采样率与配置值有较大差异")
                
                print("=========================")
        except Exception as e:
            print(f"方法1失败: {e}")
            
        # 尝试从CSV文件估算采样率
        print("\n=== 方法2: 从CSV文件估算 ===")
        estimated_fs, time_intervals = estimate_sampling_rate_from_csv()
        
        if estimated_fs:
            # 显示从CSV估算的结果
            print("\n===== CSV文件采样率估算结果 =====")
            print(f"配置文件中设置的采样率: {FS_CONFIG}Hz")
            print(f"从CSV估算的采样率: {estimated_fs:.2f}Hz")
            print(f"平均采样间隔: {1000 * np.mean(time_intervals):.2f}ms")
            print(f"采样间隔标准差: {1000 * np.std(time_intervals):.2f}ms")
            
            # 评估采样率误差
            error_percentage = abs(estimated_fs - FS_CONFIG) / FS_CONFIG * 100
            print(f"与配置值的误差: {error_percentage:.2f}%")
            
            # 判断采样率是否符合要求
            if error_percentage < 1.0:
                print("✅ 采样率符合要求 (误差 < 1%)")
            else:
                print("⚠️ 采样率与配置值有较大差异")
            
            print("===============================")
        
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        print("\n采样率验证程序已结束")

if __name__ == "__main__":
    main()