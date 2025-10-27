from pyOpenBCI import OpenBCICyton
import time
import statistics

# 配置参数
STABILIZATION_TIME = 3  # 程序启动后的稳定时间（秒）
VERIFICATION_DURATION = 1  # 采样率验证的时间窗口（秒）
TARGET_SAMPLE_RATE = 250  # 目标采样率（Hz）
TOLERANCE_PERCENTAGE = 4  # 允许的误差百分比（%）
VERIFICATION_COUNT = 10  # 要执行的验证次数

# 全局变量用于采样率验证
sample_count = 0
start_time = None
verification_start_time = None
verification_start_count = 0
verification_complete = False
verification_in_progress = False
current_verification_count = 0
sampling_rate_results = []  # 存储每次验证的采样率结果
verification_interval = 1  # 每次验证之间的间隔时间（秒）
next_verification_time = None


def print_raw(sample):
    """处理每个接收到的样本，用于采样率验证"""
    global sample_count, start_time, verification_start_time, verification_complete, \
           verification_in_progress, verification_start_count, current_verification_count, \
           sampling_rate_results, next_verification_time
    
    # 记录样本数据（可选）
    # print(sample.channels_data)
    
    # 更新样本计数
    sample_count += 1
    
    # 记录第一个样本的时间
    if sample_count == 1:
        start_time = time.time()
        print("开始接收数据...")
        print(f"接收第 {sample_count} 帧数据")
        
    # 检查是否已经过了稳定期，并且验证尚未开始
    current_time = time.time()
    if not verification_complete:
        # 第一次验证的开始条件
        if (current_verification_count == 0 and 
            current_time - start_time >= STABILIZATION_TIME and 
            not verification_in_progress):
            
            # 开始第一次采样率验证
            print(f"\n系统已稳定运行 {STABILIZATION_TIME} 秒，开始第 1/{VERIFICATION_COUNT} 次采样率验证...")
            print(f"将在接下来的 {VERIFICATION_DURATION} 秒内计算接收到的样本数")
            verification_start_time = current_time
            verification_start_count = sample_count
            verification_in_progress = True
            
        # 后续验证的开始条件
        elif (current_verification_count > 0 and 
              current_verification_count < VERIFICATION_COUNT and 
              next_verification_time is not None and 
              current_time >= next_verification_time and 
              not verification_in_progress):
            
            # 开始下一次采样率验证
            current_verification_count += 1
            print(f"\n开始第 {current_verification_count}/{VERIFICATION_COUNT} 次采样率验证...")
            print(f"将在接下来的 {VERIFICATION_DURATION} 秒内计算接收到的样本数")
            verification_start_time = current_time
            verification_start_count = sample_count
            verification_in_progress = True
        
        # 检查验证是否正在进行中并且已经过了验证时间
        elif (verification_in_progress and 
              current_time - verification_start_time >= VERIFICATION_DURATION):
            
            # 计算验证期间接收到的样本数
            verification_end_count = sample_count
            samples_in_window = verification_end_count - verification_start_count
            actual_duration = current_time - verification_start_time
            
            # 计算实际采样率
            actual_sampling_rate = samples_in_window / actual_duration if actual_duration > 0 else 0
            
            # 保存结果
            sampling_rate_results.append(actual_sampling_rate)
            
            # 显示本次验证结果
            print(f"\n=== 第 {current_verification_count + 1 if current_verification_count == 0 else current_verification_count}/{VERIFICATION_COUNT} 次采样率验证结果 ===")
            print(f"验证时间窗口: {actual_duration:.4f}秒")
            print(f"在该窗口内接收到的样本数: {samples_in_window}")
            print(f"实际采样率: {actual_sampling_rate:.2f}Hz")
            print(f"目标采样率: {TARGET_SAMPLE_RATE}Hz")
            
            # 验证采样率是否符合要求
            tolerance = TARGET_SAMPLE_RATE * (TOLERANCE_PERCENTAGE / 100)
            if abs(actual_sampling_rate - TARGET_SAMPLE_RATE) <= tolerance:
                print(f"✅ 采样率验证通过！实际采样率在允许的误差范围内 ({TARGET_SAMPLE_RATE}±{TOLERANCE_PERCENTAGE}%)")
            else:
                print(f"❌ 采样率验证失败！实际采样率与目标值偏差过大")
                print(f"偏差: {abs(actual_sampling_rate - TARGET_SAMPLE_RATE):.2f}Hz ({abs(actual_sampling_rate - TARGET_SAMPLE_RATE)/TARGET_SAMPLE_RATE*100:.2f}%)")
            
            print("=====================")
            verification_in_progress = False
            
            # 更新当前验证计数
            if current_verification_count == 0:
                current_verification_count = 1
            
            # 检查是否已完成所有验证
            if current_verification_count >= VERIFICATION_COUNT:
                # 显示所有验证的统计结果
                display_verification_summary()
                verification_complete = True
                print("\n采样率验证已全部完成，程序将继续运行。按Ctrl+C停止测试。")
            else:
                # 设置下一次验证的时间
                next_verification_time = current_time + verification_interval
                print(f"\n将在 {verification_interval} 秒后进行第 {current_verification_count + 1}/{VERIFICATION_COUNT} 次验证...")


def display_verification_summary():
    """显示所有验证的统计结果"""
    if len(sampling_rate_results) > 0:
        avg_sampling_rate = statistics.mean(sampling_rate_results)
        min_sampling_rate = min(sampling_rate_results)
        max_sampling_rate = max(sampling_rate_results)
        
        if len(sampling_rate_results) > 1:
            std_dev = statistics.stdev(sampling_rate_results)
        else:
            std_dev = 0
            
        print(f"\n=== 采样率验证总结果 ===")
        print(f"共进行了 {len(sampling_rate_results)} 次验证")
        print(f"平均采样率: {avg_sampling_rate:.2f}Hz")
        print(f"最小采样率: {min_sampling_rate:.2f}Hz")
        print(f"最大采样率: {max_sampling_rate:.2f}Hz")
        print(f"标准差: {std_dev:.2f}Hz")
        
        # 计算总体偏差
        overall_deviation = abs(avg_sampling_rate - TARGET_SAMPLE_RATE)
        deviation_percentage = (overall_deviation / TARGET_SAMPLE_RATE) * 100
        
        print(f"总体偏差: {overall_deviation:.2f}Hz ({deviation_percentage:.2f}%)")
        
        # 判断总体验证是否通过
        tolerance = TARGET_SAMPLE_RATE * (TOLERANCE_PERCENTAGE / 100)
        if overall_deviation <= tolerance:
            print(f"✅ 总体采样率验证通过！平均采样率在允许的误差范围内 ({TARGET_SAMPLE_RATE}±{TOLERANCE_PERCENTAGE}%)")
        else:
            print(f"❌ 总体采样率验证失败！平均采样率与目标值偏差过大")
        
        print(f"验证结果详情: {[round(rate, 2) for rate in sampling_rate_results]}Hz")
        print("=====================")


def main():
    """主函数，连接Cyton板并执行采样率验证"""
    global verification_complete, sampling_rate_results
    
    try:
        print("正在连接OpenBCI Cyton板...")
        board = OpenBCICyton(port='COM3', daisy=False)
        print("Cyton板连接成功！")
        
        print(f"开始数据流...")
        print(f"系统将在 {STABILIZATION_TIME} 秒稳定期后自动进行采样率验证")
        print(f"共将执行 {VERIFICATION_COUNT} 次验证，每次验证持续 {VERIFICATION_DURATION} 秒")
        print("按Ctrl+C随时停止测试")
        
        # 启动数据流
        board.start_stream(print_raw)
        
    except KeyboardInterrupt:
        print("\n正在停止测试...")
        
        # 如果验证已完成，显示总结
        if verification_complete:
            print("\n采样率验证已完成并已显示结果。")
        elif len(sampling_rate_results) > 0:
            print(f"\n测试已中断。已完成 {len(sampling_rate_results)}/{VERIFICATION_COUNT} 次验证")
            if len(sampling_rate_results) > 0:
                print(f"已收集的采样率结果: {[round(rate, 2) for rate in sampling_rate_results]}Hz")
        else:
            elapsed_time = time.time() - start_time if start_time else 0
            print(f"\n测试已停止。系统运行时间: {elapsed_time:.2f}秒，接收样本数: {sample_count}")
        
    finally:
        # 确保正确关闭数据流
        if 'board' in locals():
            try:
                board.stop_stream()
            except:
                pass
        print("Cyton板连接已关闭")


if __name__ == '__main__':
    main()

