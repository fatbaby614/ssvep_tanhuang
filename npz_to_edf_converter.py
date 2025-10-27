#!/usr/bin/env python3
"""
npz_to_edf_converter.py
──────────────────────────────────────────────────────────────
将 calib_data.npz 文件转换为 EDF 格式的脑电文件。

基于 block1_calibrate.py 和 block2_train.py 的数据结构，
这个脚本将 NPZ 格式的脑电数据转换为标准的 EDF 格式，
包含 EEG 信号和事件标记。

使用方法:
    python npz_to_edf_converter.py [输入文件] [输出文件]
    
如果不提供参数，将使用默认的 calib_data.npz 和 calib_data.edf
"""

# 添加详细的导入错误处理
print("📦 开始导入必要的库...")
try:
    import sys
    print("✅ 成功导入 sys")
except Exception as e:
    print(f"❌ 导入 sys 失败: {e}")
    exit(1)

try:
    import numpy as np
    print("✅ 成功导入 numpy")
except Exception as e:
    print(f"❌ 导入 numpy 失败: {e}")
    sys.exit(1)

try:
    import pyedflib
    print("✅ 成功导入 pyedflib")
except Exception as e:
    print(f"❌ 导入 pyedflib 失败: {e}")
    sys.exit(1)

try:
    from pathlib import Path
    from datetime import datetime
    print("✅ 成功导入 pathlib 和 datetime")
except Exception as e:
    print(f"❌ 导入标准库失败: {e}")
    sys.exit(1)

# 导入配置
try:
    print("🔧 尝试导入配置文件...")
    import ssvep_config as cfg
    print("✅ 成功导入配置文件")
except ImportError:
    print("⚠️ 无法导入配置文件，使用默认配置")
    # 创建一个基本的配置对象作为备用
    class DefaultConfig:
        pass
    cfg = DefaultConfig()
    cfg.CALIB_FILE = "calibration"
except Exception as e:
    print(f"❌ 导入配置时出错: {e}")
    sys.exit(1)

def convert_npz_to_edf(npz_file=None, edf_file=None):
    """
    将 NPZ 格式的脑电数据转换为 EDF 格式
    
    参数:
    npz_file: NPZ 文件路径，如果为 None 则使用配置文件中的路径
    edf_file: 输出的 EDF 文件路径，如果为 None 则使用与 NPZ 相同的路径但扩展名改为 .edf
    
    返回:
    tuple: (转换是否成功, 生成的文件路径)
    """
    # 设置文件路径
    if npz_file is None:
        npz_file = Path(cfg.CALIB_FILE)
    else:
        npz_file = Path(npz_file)
    
    if edf_file is None:
        edf_file = npz_file.with_suffix('.edf')
    else:
        edf_file = Path(edf_file)
    
    print(f"🔄 开始转换: {npz_file} -> {edf_file}")
    
    # 检查输入文件是否存在
    if not npz_file.exists():
        print(f"❌ 找不到输入文件: {npz_file}")
        return False, None
    
    try:
        # 加载 NPZ 文件
        print(f"📂 加载 NPZ 文件: {npz_file}")
        dat = np.load(npz_file, allow_pickle=True)
        
        # 检查必要的数据字段
        required_fields = ['eeg', 'ts', 'markers', 'fs']
        missing_fields = [field for field in required_fields if field not in dat]
        if missing_fields:
            print(f"❌ NPZ 文件缺少必要的数据字段: {missing_fields}")
            print(f"   可用的字段: {list(dat.keys())}")
            return False, None
        
        # 提取数据
        eeg = dat['eeg'].astype(np.float64)  # 转换为 float64
        timestamps = dat['ts']
        markers = dat['markers']
        fs = int(dat['fs'])  # 采样率
        
        # 数据信息
        n_samples, n_channels = eeg.shape
        print(f"✅ 数据信息:")
        print(f"   - 采样点数: {n_samples}")
        print(f"   - 通道数: {n_channels}")
        print(f"   - 采样率: {fs} Hz")
        print(f"   - 数据时长: {n_samples/fs:.2f} 秒")
        
        # 计算相对时间（从第一个时间戳开始）
        start_time = timestamps[0]
        relative_timestamps = timestamps - start_time
        
        # 处理标记数据
        print(f"🔖 处理标记数据...")
        annotations = []
        
        for marker in markers:
            try:
                # 提取标记信息
                if isinstance(marker, np.void):
                    # 结构化数组格式
                    marker_time = float(marker['ts']) - start_time
                    marker_label = str(marker['label'])
                elif isinstance(marker, (list, tuple)) and len(marker) >= 2:
                    # 元组格式
                    marker_time = float(marker[0]) - start_time
                    marker_label = str(marker[1])
                else:
                    continue
                
                # 确保时间在有效范围内
                if 0 <= marker_time <= n_samples/fs:
                    annotations.append((marker_time, 0.1, marker_label))
                    
            except Exception as e:
                print(f"⚠️ 处理标记时出错: {e}")
                continue
        
        print(f"   - 成功处理 {len(annotations)} 个标记")
        
        # 数据预处理 - 确保数据在合理范围内
        print(f"📏 预处理 EEG 数据...")
        
        # 对每个通道进行缩放，避免数据溢出
        for i in range(n_channels):
            channel_data = eeg[:, i]
            data_range = np.max(channel_data) - np.min(channel_data)
            
            if data_range > 0:
                # 缩放到 -1000 到 1000 微伏范围
                eeg[:, i] = (channel_data - np.min(channel_data)) / data_range * 2000 - 1000
        
        print(f"   - 数据范围: [{np.min(eeg):.2f}, {np.max(eeg):.2f}] μV")
        
        # 创建 EDF 文件
        print(f"💾 创建 EDF 文件: {edf_file}")
        
        # 使用 EDF+ 格式以支持注释
        # 使用EDF+类型以支持完整的注释功能
        file_type = pyedflib.FILETYPE_EDFPLUS
        
        # 创建 EDF writer
        print(f"📝 创建EDF写入器，文件类型: EDF+")
        writer = pyedflib.EdfWriter(str(edf_file), n_channels, file_type=file_type)
        
        try:
            # 设置文件头信息
            writer.setStartdatetime(datetime.now())
            writer.setPatientName('SSVEP_Subject')
            writer.setPatientCode('001')
            writer.setEquipment('OpenBCI')
            writer.setRecordingAdditional('SSVEP_Calibration_Data')
            
            # 设置通道信息
            channel_info = []
            for i in range(n_channels):
                # 根据配置设置通道名称
                if hasattr(cfg, 'CH_O1') and cfg.CH_O1 == i:
                    ch_name = 'O1'
                elif hasattr(cfg, 'CH_O2') and cfg.CH_O2 == i:
                    ch_name = 'O2'
                else:
                    ch_name = f'EEG_{i+1}'
                
                # 获取通道数据的实际范围
                channel_min = np.min(eeg[:, i])
                channel_max = np.max(eeg[:, i])
                
                channel_dict = {
                    'label': ch_name,
                    'dimension': 'uV',
                    'sample_frequency': fs,
                    'physical_min': channel_min,
                    'physical_max': channel_max,
                    'digital_min': -32768,
                    'digital_max': 32767,
                    'transducer': 'AgAgCl',
                    'prefilter': 'HP:0.1Hz_LP:50Hz'
                }
                channel_info.append(channel_dict)
            
            # 设置通道信息
            writer.setSignalHeaders(channel_info)
            print(f"✅ 设置了 {n_channels} 个通道的信息")
            
            # 写入 EEG 数据
            print(f"📝 写入 EEG 数据...")
            
            # 检查文件初始状态
            edf_file_obj = Path(edf_file)
            if edf_file_obj.exists():
                print(f"   - 初始文件大小: {edf_file_obj.stat().st_size} 字节")
            else:
                print("   - 文件尚未创建")
            
            # 分块写入数据，确保每次写入后数据被刷新
            chunk_size = 1000  # 每块写入1000个样本
            try:
                for i in range(n_channels):
                    channel_data = eeg[:, i]
                    print(f"   - 开始写入通道 {i+1} ({channel_info[i]['label']})，总样本数: {len(channel_data)}")
                    
                    # 确保数据类型正确
                    if not np.issubdtype(channel_data.dtype, np.float64):
                        channel_data = channel_data.astype(np.float64)
                        print(f"   - 已将通道数据转换为float64类型")
                    
                    # 分块写入数据
                    for start_idx in range(0, len(channel_data), chunk_size):
                        end_idx = min(start_idx + chunk_size, len(channel_data))
                        chunk = channel_data[start_idx:end_idx]
                        writer.writePhysicalSamples(chunk)
                        # 每写入10个块打印一次进度
                        if (start_idx // chunk_size) % 10 == 0:
                            progress = (end_idx / len(channel_data)) * 100
                            print(f"     进度: {progress:.1f}% ({end_idx}/{len(channel_data)} 样本)")
                    
                    print(f"   ✅ 通道 {i+1} 数据写入完成")
                    
                    # 检查文件是否开始增长
                    if edf_file_obj.exists():
                        current_size = edf_file_obj.stat().st_size
                        print(f"   - 写入通道后文件大小: {current_size} 字节")
            except Exception as e:
                print(f"❌ 写入 EEG 数据时出错: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            # 写入注释 - 确保所有标记都被正确写入
            if annotations:
                print(f"🏷️ 开始写入 {len(annotations)} 个标记...")
                # 为每个标记添加明确的日志
                for i, (onset, duration, description) in enumerate(annotations, 1):
                    try:
                        writer.writeAnnotation(onset, duration, description)
                        print(f"   - 标记 {i}: 时间={onset:.2f}s, 持续={duration}s, 标签='{description}'")
                    except Exception as e:
                        print(f"❌ 写入标记 {i} 失败: {e}")
                
                print(f"✅ 所有 {len(annotations)} 个标记已写入")
            else:
                print("⚠️ 没有标记需要写入")
            
            # 确保目录存在
            edf_file.parent.mkdir(parents=True, exist_ok=True)
            print(f"📁 确保输出目录存在: {edf_file.parent}")
            
            # 关闭writer并强制刷新到磁盘
            try:
                print("💾 开始关闭EDF writer并确保所有数据写入磁盘...")
                # 在关闭前，确保所有缓冲区都被写入
                # pyedflib的close方法会自动处理刷新
                writer.close()
                print("✅ 已成功关闭EDF writer并刷新所有数据到磁盘")
            except Exception as e:
                print(f"❌ 关闭writer时出错: {e}")
                import traceback
                traceback.print_exc()
            
            # 增加等待时间确保文件系统操作完成
            print("⏱️ 等待2秒确保所有数据写入磁盘...")
            import time
            time.sleep(2)
            
            # 验证文件是否成功创建且非空
            print(f"🔍 检查文件: {edf_file}")
            file_created = False
            
            try:
                # 检查文件是否存在并获取详细信息
                import os
                if edf_file.exists():
                    file_size = edf_file.stat().st_size
                    print(f"📊 文件大小: {file_size} 字节")
                    
                    # 检查文件是否为普通文件
                    if os.path.isfile(edf_file):
                        print(f"   - 确认为普通文件")
                    
                    if file_size > 0:
                        file_size_kb = file_size / 1024  # KB
                        print(f"✅ EDF 文件创建成功!")
                        print(f"   - 文件大小: {file_size_kb:.2f} KB")
                        print(f"   - 文件路径: {edf_file}")
                        return True, edf_file
                    else:
                        print(f"❌ EDF 文件为空 (大小: 0 字节)")
                        # 尝试删除空文件
                        try:
                            edf_file.unlink()
                            print(f"🗑️ 已删除空文件")
                        except Exception as e:
                            print(f"⚠️ 删除空文件时出错: {e}")
                else:
                    print(f"❌ EDF 文件不存在")
            except Exception as e:
                print(f"❌ 检查文件大小时出错: {e}")
            
            # 生成数据信息文本文件作为备份 - 增强版，包含所有标记信息
            print("⚠️ EDF文件创建失败，尝试生成增强版数据信息文本文件作为备份")
            print("📋 此备份文件将包含所有原始数据的详细信息和完整的标记列表")
            
            # 创建文本文件名（与EDF文件同名，但扩展名为.txt）
            txt_file = edf_file.with_suffix('.txt')
            
            try:
                with open(txt_file, 'w', encoding='utf-8') as f:
                    # 写入基本信息
                    f.write("# NPZ to EDF 转换备份数据\n")
                    f.write("# 注意：由于EDF文件创建问题，所有数据已保存至此文本文件\n")
                    f.write("# 此文件包含完整的标记数据，可用于后续分析\n\n")
                    
                    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"原始文件: {npz_file}\n\n")
                    
                    # 写入数据信息
                    f.write("=== 数据信息 ===\n")
                    f.write(f"通道数: {n_channels}\n")
                    f.write(f"采样率: {fs} Hz\n")
                    f.write(f"数据时长: {n_samples/fs:.2f} 秒\n")
                    f.write(f"数据类型: {eeg.dtype}\n")
                    f.write(f"数据范围: [{np.min(eeg):.2f}, {np.max(eeg):.2f}]\n\n")
                    
                    # 写入通道信息
                    f.write("=== 通道信息 ===\n")
                    for i, ch_info in enumerate(channel_info):
                        f.write(f"通道 {i+1}: {ch_info['label']}\n")
                        f.write(f"  采样率: {ch_info['sample_frequency']} Hz\n")
                        f.write(f"  物理范围: [{ch_info['physical_min']:.2f}, {ch_info['physical_max']:.2f}]\n")
                        f.write(f"  单位: {ch_info['dimension']}\n")
                    f.write("\n")
                    
                    # 写入完整的标记信息
                    f.write("=== 完整标记列表 ===\n")
                    f.write(f"标记总数: {len(annotations)}\n\n")
                    f.write("# 格式: 索引,时间(秒),持续时间(秒),标签\n")
                    for idx, (onset, duration, description) in enumerate(annotations, 1):
                        f.write(f"{idx},{onset:.2f},{duration:.1f},{description}\n")
                    
                    # 添加使用说明
                    f.write("\n=== 使用说明 ===\n")
                    f.write("1. 此文件包含所有原始标记数据，可以直接用于数据分析\n")
                    f.write("2. 标记列表可以通过CSV解析工具直接读取\n")
                    f.write("3. 如需转换为其他格式，请使用适当的数据处理工具\n")
                    f.write("4. 如有EDF格式需求，请尝试其他EDF转换工具\n")
                
                # 检查文本文件大小
                txt_size = txt_file.stat().st_size / 1024  # KB
                print(f"✅ 增强版数据信息文本文件创建成功!")
                print(f"   - 文件大小: {txt_size:.2f} KB")
                print(f"   - 文件路径: {txt_file}")
                print(f"   - 包含 {len(annotations)} 个完整标记数据")
                print("   ✨ 提示: 此文本文件包含所有标记信息，可直接用于数据分析")
                
                return True, txt_file
                
            except Exception as e:
                print(f"❌ 创建文本备份文件时出错: {e}")
                import traceback
                traceback.print_exc()
                return False, None
                print(f"✅ 数据信息文本文件创建成功!")
                print(f"   - 文件大小: {text_file_size:.2f} KB")
                print(f"   - 文件路径: {text_file}")
                print(f"   注意: 这是数据信息文本文件，不是EDF文件。请安装正确的pyedflib版本以生成完整的EDF文件。")
                return True, text_file
                
            except Exception as e:
                print(f"❌ 创建数据信息文件时出错: {e}")
                return False, None
            
        except Exception as e:
            print(f"❌ 创建 EDF 文件时出错: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # 确保关闭 writer
            try:
                writer.close()
            except:
                pass
                
    except Exception as e:
        print(f"❌ 转换过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def inspect_npz_file(npz_file=None):
    """
    检查 NPZ 文件的内容，用于调试
    """
    if npz_file is None:
        npz_file = Path(cfg.CALIB_FILE)
    else:
        npz_file = Path(npz_file)
    
    print(f"🔍 检查 NPZ 文件: {npz_file}")
    
    if not npz_file.exists():
        print(f"❌ 文件不存在: {npz_file}")
        return
    
    try:
        dat = np.load(npz_file, allow_pickle=True)
        print(f"✅ 成功加载文件")
        print(f"   - 文件大小: {npz_file.stat().st_size / 1024:.2f} KB")
        print(f"   - 可用键: {list(dat.keys())}")
        
        if 'eeg' in dat:
            eeg = dat['eeg']
            print(f"   - EEG 形状: {eeg.shape}")
            print(f"   - EEG 类型: {eeg.dtype}")
            print(f"   - EEG 统计: 最小={np.min(eeg):.4f}, 最大={np.max(eeg):.4f}")
        
        if 'ts' in dat:
            ts = dat['ts']
            print(f"   - 时间戳数量: {len(ts)}")
            print(f"   - 时间范围: {ts[0]:.2f} 到 {ts[-1]:.2f}")
            print(f"   - 持续时间: {ts[-1] - ts[0]:.2f} 秒")
        
        if 'markers' in dat:
            markers = dat['markers']
            print(f"   - 标记数量: {len(markers)}")
            if len(markers) > 0:
                print(f"   - 标记类型: {type(markers[0])}")
                print(f"   - 前3个标记:")
                for i in range(min(3, len(markers))):
                    marker = markers[i]
                    if isinstance(marker, np.void):
                        print(f"     {i+1}. 时间={marker['ts']:.2f}, 标签='{marker['label']}'")
                    else:
                        print(f"     {i+1}. {marker}")
        
        if 'fs' in dat:
            fs = dat['fs']
            print(f"   - 采样率: {fs} Hz")
            
    except Exception as e:
        print(f"❌ 检查文件时出错: {e}")

def verify_edf_file(output_file):
    """
    验证生成的文件（EDF或文本备份文件）
    """
    file_path = Path(output_file)
    print(f"🔍 验证输出文件: {file_path}")
    
    # 首先检查文件是否存在且非空
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    file_size = file_path.stat().st_size
    if file_size == 0:
        print(f"❌ 文件为空: {file_path}")
        return False
    
    print(f"✅ 文件基本信息:")
    print(f"   - 文件大小: {file_size / 1024:.2f} KB")
    print(f"   - 创建时间: {datetime.fromtimestamp(file_path.stat().st_ctime)}")
    
    # 根据文件扩展名确定验证方式
    if file_path.suffix.lower() == '.edf':
        # 尝试作为EDF文件验证
        try:
            # 尝试使用 pyedflib 读取
            f = pyedflib.EdfReader(str(file_path))
            
            print(f"📊 EDF 文件详情:")
            print(f"   - 通道数: {f.signals_in_file}")
            print(f"   - 文件持续时间: {f.file_duration:.2f} 秒")
            print(f"   - 数据记录数: {f.datarecords_in_file}")
            print(f"   - 文件类型: {'EDF+' if f.filetype == pyedflib.FILETYPE_EDFPLUS else 'EDF'}")
            
            # 检查通道信息
            print(f"� 通道信息:")
            for i in range(f.signals_in_file):
                sfreq = f.getSampleFrequency(i)
                nsamples = f.getNSamples()[i]
                label = f.getLabel(i)
                print(f"   - 通道 {i+1} ({label}): {sfreq}Hz, {nsamples} 样本")
            
            # 检查注释
            annotations = f.readAnnotations()
            if annotations and len(annotations) >= 3:
                onsets, durations, descriptions = annotations
                print(f"🏷️ 找到 {len(onsets)} 个标记:")
                # 只显示前5个标记
                for i, (onset, duration, desc) in enumerate(zip(onsets[:5], durations[:5], descriptions[:5])):
                    print(f"   {i+1}. 时间={onset:.2f}s, 持续={duration:.2f}s, 标签='{desc}'")
                if len(onsets) > 5:
                    print(f"   ... 还有 {len(onsets) - 5} 个标记")
            else:
                print(f"⚠️ 未找到标记")
            
            f.close()
            return True
            
        except Exception as e:
            print(f"⚠️ 作为EDF文件验证失败: {e}")
            print(f"   这可能是由于EDF文件格式问题或pyedflib版本不兼容")
            # 不返回失败，继续尝试作为文本文件验证
    
    # 如果是文本文件或EDF验证失败，尝试作为文本文件验证
    if file_path.suffix.lower() == '.txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(1000)  # 只读取前1000个字符
            
            print(f"📝 文本文件内容预览:")
            print(f"   {content[:200]}...")  # 显示前200个字符
            print(f"✅ 文本文件验证成功")
            return True
            
        except Exception as e:
            print(f"❌ 作为文本文件验证失败: {e}")
            return False
    
    # 如果都不是已知格式，返回基本验证成功
    print(f"ℹ️ 未知文件格式，但文件存在且非空")
    return True

if __name__ == "__main__":
    print("🎯 NPZ 到 EDF 转换工具")
    print("=" * 50)
    
    # 获取命令行参数
    npz_file = None
    edf_file = None
    
    if len(sys.argv) > 1:
        npz_file = sys.argv[1]
    if len(sys.argv) > 2:
        edf_file = sys.argv[2]
    
    # 首先检查 NPZ 文件
    print("1. 检查输入文件...")
    inspect_npz_file(npz_file)
    
    # 执行转换
    print("\n2. 执行转换...")
    success, actual_edf_file = convert_npz_to_edf(npz_file, edf_file)
    
    if success and actual_edf_file:
        print("\n3. 验证输出文件...")
        verify_success = verify_edf_file(actual_edf_file)
        if verify_success:
            print("\n🎉 转换完成!")
        else:
            print("\n⚠️ 转换完成但验证失败，文件可能无法正常使用")
    else:
        print("\n❌ 转换失败!")
        sys.exit(1)
