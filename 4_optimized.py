import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import joblib
from numba import jit, prange
import time
import os

# 设置CPU优化环境变量
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_count())
os.environ['MKL_NUM_THREADS'] = str(cpu_count())
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_count())

# 更安全的GPU检测
GPU_AVAILABLE = False
try:
    import cupy as cp
    # 检查是否真的有可用的GPU设备
    device_count = cp.cuda.runtime.getDeviceCount()
    if device_count > 0:
        # 测试GPU基本功能
        test_array = cp.array([1, 2, 3])
        cp.asnumpy(test_array)
        GPU_AVAILABLE = True
        print(f"检测到GPU: {device_count}张")
    else:
        GPU_AVAILABLE = False
        print("CUDA运行时可用但未检测到GPU设备")
except Exception as e:
    import numpy as cp
    GPU_AVAILABLE = False
    print(f"GPU不可用: {str(e)[:100]}...")
    print("使用CPU多核并行计算")

# -------------------------- 1. 常量与初始参数定义 --------------------------
g = 9.80665  # 重力加速度 (m/s²)
epsilon = 1e-12  # 数值计算保护阈值

# 目标定义
fake_target = np.array([0.0, 0.0, 0.0])  # 假目标（原点）
real_target = {
    "center": np.array([0.0, 200.0, 0.0]),  # 底面圆心
    "r": 7.0,  # 圆柱半径
    "h": 10.0   # 圆柱高度
}

# 无人机初始位置
uav_positions = {
    "FY1": np.array([17800.0, 0.0, 1800.0]),
    "FY2": np.array([12000.0, 1400.0, 1400.0]),
    "FY3": np.array([6000.0, -3000.0, 700.0]),
    "FY4": np.array([11000.0, 2000.0, 1800.0]),
    "FY5": np.array([13000.0, -2000.0, 1300.0])
}

# 烟幕参数
smoke_param = {
    "r": 10.0,  # 有效半径(m)
    "sink_speed": 3.0,  # 起爆后下沉速度(m/s)
    "valid_time": 20.0  # 有效遮蔽时间(s)
}

# 导弹M1参数
missile_m1 = {
    "init_pos": np.array([20000.0, 0.0, 2000.0]),  # 初始位置
    "speed": 300.0  # 飞行速度(m/s)
}

dt = 0.001  # 时间步长

# -------------------------- 2. Numba加速核心计算函数 --------------------------
@jit(nopython=True, cache=True)
def calc_uav_direction_numba(angle_deg):
    """Numba加速的无人机飞行方向计算"""
    angle_rad = np.radians(angle_deg)
    return np.array([np.cos(angle_rad), np.sin(angle_rad)])

@jit(nopython=True, cache=True)
def calc_drop_point_numba(uav_init_pos, uav_speed, drop_delay, angle_deg):
    """Numba加速的投放点计算"""
    dir_vec_xy = calc_uav_direction_numba(angle_deg)
    flight_dist = uav_speed * drop_delay
    drop_xy = uav_init_pos[:2] + dir_vec_xy * flight_dist
    return np.array([drop_xy[0], drop_xy[1], uav_init_pos[2]])

@jit(nopython=True, cache=True)
def calc_det_point_numba(drop_point, uav_speed, det_delay, angle_deg):
    """Numba加速的起爆点计算"""
    dir_vec_xy = calc_uav_direction_numba(angle_deg)
    horizontal_dist = uav_speed * det_delay
    det_xy = drop_point[:2] + dir_vec_xy * horizontal_dist
    drop_h = 0.5 * g * det_delay ** 2
    det_z = drop_point[2] - drop_h
    return np.array([det_xy[0], det_xy[1], det_z])

@jit(nopython=True, cache=True)
def is_segment_intersect_sphere_numba(M, P, C, r):
    """Numba加速的线段-球相交检测"""
    MP = P - M
    MC = C - M
    
    a = np.dot(MP, MP)
    if a < epsilon:
        return np.linalg.norm(MC) <= r + epsilon
    
    b = -2 * np.dot(MP, MC)
    c = np.dot(MC, MC) - r ** 2
    
    discriminant = b ** 2 - 4 * a * c
    if discriminant < -epsilon:
        return False
    
    if discriminant < 0:
        discriminant = 0
    
    sqrt_d = np.sqrt(discriminant)
    s1 = (-b - sqrt_d) / (2 * a)
    s2 = (-b + sqrt_d) / (2 * a)
    
    return (s1 <= 1.0 + epsilon) and (s2 >= -epsilon)

@jit(nopython=True, cache=True)
def check_all_samples_shielded_numba(missile_pos, smoke_center, smoke_r, target_samples):
    """Numba加速的批量遮蔽检测"""
    for i in range(len(target_samples)):
        if not is_segment_intersect_sphere_numba(missile_pos, target_samples[i], smoke_center, smoke_r):
            return False
    return True

@jit(nopython=True, cache=True)
def check_any_smoke_shields_numba(missile_pos, smoke_centers, smoke_r, target_samples):
    """检查是否有任一烟幕完全遮蔽目标"""
    for i in range(len(smoke_centers)):
        if check_all_samples_shielded_numba(missile_pos, smoke_centers[i], smoke_r, target_samples):
            return True
    return False

# -------------------------- 3. 核心计算函数 --------------------------
def calc_uav_direction(uav_init_pos, fake_target, angle_deg):
    """计算无人机飞行方向向量（使用绝对角度）"""
    return calc_uav_direction_numba(angle_deg)

def calc_drop_point(uav_init_pos, uav_speed, drop_delay, fake_target, angle_deg):
    """计算烟幕弹投放点"""
    return calc_drop_point_numba(uav_init_pos, uav_speed, drop_delay, angle_deg)

def calc_det_point(drop_point, uav_speed, det_delay, g, fake_target, angle_deg, uav_init_pos):
    """计算烟幕弹起爆点"""
    return calc_det_point_numba(drop_point, uav_speed, det_delay, angle_deg)

def generate_high_density_samples(target, num_circle=60, num_height=20):
    """生成高密度采样点"""
    samples = []
    center = target["center"]
    r = target["r"]
    h = target["h"]
    center_xy = center[:2]
    min_z = center[2]
    max_z = center[2] + h
    
    # 外表面采样
    theta = np.linspace(0, 2*np.pi, num_circle, endpoint=False)
    for th in theta:
        x = center_xy[0] + r * np.cos(th)
        y = center_xy[1] + r * np.sin(th)
        samples.append([x, y, min_z])
        samples.append([x, y, max_z])
    
    heights = np.linspace(min_z, max_z, num_height, endpoint=True)
    for z in heights:
        for th in theta:
            x = center_xy[0] + r * np.cos(th)
            y = center_xy[1] + r * np.sin(th)
            samples.append([x, y, z])
    
    # 内部网格点采样
    radii = np.linspace(0, r, 5, endpoint=True)
    inner_heights = np.linspace(min_z, max_z, 10, endpoint=True)
    inner_thetas = np.linspace(0, 2*np.pi, 12, endpoint=False)
    
    for z in inner_heights:
        for rad in radii:
            for th in inner_thetas:
                x = center_xy[0] + rad * np.cos(th)
                y = center_xy[1] + rad * np.sin(th)
                samples.append([x, y, z])
    
    return np.unique(np.array(samples), axis=0)

# -------------------------- 4. 高性能CPU并行版本 --------------------------
def calc_total_shield_time_optimized(params, uav_names, target_samples):
    """优化的遮蔽时间计算 - 单线程版本，避免pickle问题"""
    try:
        num_uavs = len(uav_names)
        
        # 解析参数
        uav_params = {}
        for i, uav_name in enumerate(uav_names):
            idx = i * 4
            uav_params[uav_name] = {
                "speed": params[idx],
                "drop_delay": params[idx + 1],
                "det_delay": params[idx + 2],
                "angle": params[idx + 3]
            }
        
        # 预计算所有无人机数据
        uav_det_points = []
        uav_det_times = []
        
        for uav_name in uav_names:
            uav_pos = uav_positions[uav_name]
            param = uav_params[uav_name]
            
            drop_point = calc_drop_point_numba(uav_pos, param["speed"], param["drop_delay"], param["angle"])
            det_point = calc_det_point_numba(drop_point, param["speed"], param["det_delay"], param["angle"])
            det_time = param["drop_delay"] + param["det_delay"]
            
            uav_det_points.append(det_point)
            uav_det_times.append(det_time)
        
        # 转换为numpy数组
        uav_det_points = np.array(uav_det_points)
        uav_det_times = np.array(uav_det_times)
        
        # 计算导弹飞行方向
        missile_vec = fake_target - missile_m1["init_pos"]
        missile_dist = np.linalg.norm(missile_vec)
        if missile_dist < epsilon:
            missile_dir = np.array([0.0, 0.0, 0.0])
        else:
            missile_dir = missile_vec / missile_dist
        
        # 确定时间范围
        min_det_time = np.min(uav_det_times)
        max_det_time = np.max(uav_det_times)
        
        t_start = min_det_time
        t_end = max_det_time + smoke_param["valid_time"]
        t_list = np.arange(t_start, t_end + dt, dt)
        
        # 单线程计算，但使用Numba加速
        valid_total = 0.0
        
        for t in t_list:
            # 计算导弹位置
            missile_pos = missile_m1["init_pos"] + missile_dir * missile_m1["speed"] * t
            
            # 计算有效烟幕数量
            active_smoke_centers = []
            
            for i, uav_name in enumerate(uav_names):
                det_time = uav_det_times[i]
                det_point = uav_det_points[i]
                
                if t >= det_time and t <= det_time + smoke_param["valid_time"]:
                    sink_time = t - det_time
                    smoke_center = np.array([
                        det_point[0],
                        det_point[1],
                        det_point[2] - smoke_param["sink_speed"] * sink_time
                    ])
                    active_smoke_centers.append(smoke_center)
            
            # 检查遮蔽
            if active_smoke_centers:
                active_smoke_centers = np.array(active_smoke_centers)
                if check_any_smoke_shields_numba(missile_pos, active_smoke_centers, smoke_param["r"], target_samples):
                    valid_total += dt
        
        return valid_total
        
    except Exception as e:
        print(f"计算错误: {e}")
        return 0.0

# 全局目标函数，避免pickle问题
def global_objective(params):
    """全局目标函数，可以被pickle序列化"""
    global current_uav_names, current_target_samples
    return -calc_total_shield_time_optimized(params, current_uav_names, current_target_samples)

# -------------------------- 5. 智能优化策略 --------------------------
def ultimate_optimization_strategy(uav_names, target_samples):
    """CPU优化策略 - 修复pickle问题"""
    
    # 设置全局变量，避免pickle问题
    global current_uav_names, current_target_samples
    current_uav_names = uav_names
    current_target_samples = target_samples
    
    print("=== 服务器资源检测 ===")
    total_cores = cpu_count()
    print(f"CPU核心数: {total_cores}")
    print("GPU不可用，使用CPU多核并行计算")
    
    n_samples = len(target_samples)
    n_params = len(uav_names) * 4
    
    # 根据CPU核心数选择策略
    if total_cores >= 100:
        print("策略: 超高度CPU并行 (大型服务器)")
        n_workers = min(40, int(total_cores * 0.4))  # 减少workers数量
    elif total_cores >= 50:
        print("策略: 高度CPU并行")
        n_workers = min(20, int(total_cores * 0.4))
    elif total_cores >= 16:
        print("策略: 中度CPU并行")
        n_workers = min(8, total_cores // 2)
    else:
        print("策略: 轻度并行")
        n_workers = max(1, total_cores // 4)
    
    print(f"优化使用CPU核心数: {n_workers}")
    
    # 参数边界
    bounds = [(70, 140), (0.1, 5.0), (0.1, 5.0), (0, 360)] * len(uav_names)
    
    print(f"\n=== 开始多阶段优化 ===")
    print(f"优化参数数量: {len(bounds)}")
    print(f"目标采样点数: {n_samples}")
    
    start_time = time.time()
    
    # 阶段1: 快速全局搜索
    print("\n阶段1: 快速全局搜索...")
    result1 = differential_evolution(
        global_objective,
        bounds,
        maxiter=20,  # 减少迭代次数
        popsize=10,  # 减少种群大小
        workers=min(n_workers, 8),  # 限制workers数量
        seed=42,
        disp=True,
        updating='deferred'  # 显式设置更新策略
    )
    
    stage1_time = time.time() - start_time
    print(f"阶段1完成，用时: {stage1_time:.2f}秒，最优值: {-result1.fun:.6f}")
    
    # 阶段2: 精细搜索
    print("\n阶段2: 精细搜索...")
    
    # 缩小搜索范围
    best_params = result1.x
    refined_bounds = []
    for i, (low, high) in enumerate(bounds):
        center = best_params[i]
        range_size = (high - low) * 0.15  # 缩小到15%范围
        new_low = max(low, center - range_size/2)
        new_high = min(high, center + range_size/2)
        refined_bounds.append((new_low, new_high))
    
    result2 = differential_evolution(
        global_objective,
        refined_bounds,
        maxiter=50,  # 精细搜索更多迭代
        popsize=15,
        workers=min(n_workers, 12),
        seed=43,
        polish=True,
        disp=True,
        updating='deferred'
    )
    
    total_time = time.time() - start_time
    
    # 选择最优结果
    if result2.fun < result1.fun:
        final_result = result2
        print(f"\n精细搜索找到更优解: {-result2.fun:.6f}")
    else:
        final_result = result1
        print(f"\n初始搜索结果更优: {-result1.fun:.6f}")
    
    print(f"总优化时间: {total_time:.2f}秒")
    print(f"函数评估次数: {result1.nfev + result2.nfev}")
    
    return final_result

def optimize_multi_uav(uav_names, target_samples):
    """优化多架无人机参数 - 主要接口函数"""
    return ultimate_optimization_strategy(uav_names, target_samples)

# -------------------------- 6. 主程序 --------------------------
if __name__ == "__main__":
    print("=== 多无人机协同烟幕干扰优化系统 (CPU高性能版本) ===")
    
    # 生成目标采样点
    print("\n生成真目标采样点...")
    target_samples = generate_high_density_samples(real_target, num_circle=40, num_height=12)  # 进一步减少采样点
    print(f"生成真目标采样点：{len(target_samples)}个")
    
    # 选择参与任务的无人机
    uav_names = ["FY1", "FY2", "FY3"]
    
    print(f"\n开始优化{len(uav_names)}架无人机的协同干扰策略...")
    print("参与无人机：", uav_names)
    for name in uav_names:
        print(f"{name}: {uav_positions[name]}")
    
    # 执行优化
    print(f"\n目标导弹: M1 位置 {missile_m1['init_pos']}, 速度 {missile_m1['speed']} m/s")
    
    start_total = time.time()
    result = optimize_multi_uav(uav_names, target_samples)
    total_optimization_time = time.time() - start_total
    
    if result.success:
        print(f"\n=== 优化成功！===")
        print(f"最大遮蔽时间：{-result.fun:.6f} 秒")
        print(f"总优化时间：{total_optimization_time:.2f} 秒")
        print(f"函数评估次数：{result.nfev}")
        print(f"平均每次评估：{total_optimization_time/result.nfev:.4f} 秒")
        
        # 解析最优参数
        optimal_params = result.x
        results_data = []
        
        print(f"\n=== 详细优化结果 ===")
        for i, uav_name in enumerate(uav_names):
            idx = i * 4
            speed = optimal_params[idx]
            drop_delay = optimal_params[idx + 1]
            det_delay = optimal_params[idx + 2]
            angle = optimal_params[idx + 3]
            
            # 计算投放点和起爆点
            uav_pos = uav_positions[uav_name]
            
            drop_point = calc_drop_point(
                uav_pos, speed, drop_delay, fake_target, angle
            )
            
            det_point = calc_det_point(
                drop_point, speed, det_delay, g,
                fake_target, angle, uav_pos
            )
            
            # 计算单架无人机的贡献时间
            single_shield_time = calc_total_shield_time_optimized(
                [speed, drop_delay, det_delay, angle], 
                [uav_name], 
                target_samples
            )
            
            results_data.append({
                "无人机编号": uav_name,
                "无人机运动方向": f"{angle:.2f}",
                "无人机运动速度 (m/s)": f"{speed:.2f}",
                "烟幕干扰弹投放点的x坐标 (m)": f"{drop_point[0]:.2f}",
                "烟幕干扰弹投放点的y坐标 (m)": f"{drop_point[1]:.2f}",
                "烟幕干扰弹投放点的z坐标 (m)": f"{drop_point[2]:.2f}",
                "烟幕干扰弹起爆点的x坐标 (m)": f"{det_point[0]:.2f}",
                "烟幕干扰弹起爆点的y坐标 (m)": f"{det_point[1]:.2f}",
                "烟幕干扰弹起爆点的z坐标 (m)": f"{det_point[2]:.2f}",
                "有效干扰时长 (s)": f"{single_shield_time:.4f}"
            })
            
            print(f"\n{uav_name} 优化结果：")
            print(f"  飞行角度：{angle:.6f}°")
            print(f"  飞行速度：{speed:.6f} m/s")
            print(f"  投放延迟：{drop_delay:.6f} s")
            print(f"  起爆延迟：{det_delay:.6f} s")
            print(f"  投放点：({drop_point[0]:.4f}, {drop_point[1]:.4f}, {drop_point[2]:.4f})")
            print(f"  起爆点：({det_point[0]:.4f}, {det_point[1]:.4f}, {det_point[2]:.4f})")
            print(f"  单独贡献时间：{single_shield_time:.4f} s")
        
        # 保存到Excel文件
        try:
            df = pd.DataFrame(results_data)
            output_file = "result2.xlsx"
            df.to_excel(output_file, index=False)
            print(f"\n结果已保存到 {output_file}")
        except Exception as e:
            print(f"\n保存Excel文件失败: {e}")
            # 保存为CSV作为备选
            try:
                df.to_csv("result2.csv", index=False)
                print("已保存为CSV格式: result2.csv")
            except Exception as e2:
                print(f"CSV保存也失败: {e2}")
        
        print(f"\n*** 协同总遮蔽时间：{-result.fun:.6f} 秒 ***")
        
    else:
        print("优化失败:", result.message)

    print(f"\n程序执行完毕，总用时: {time.time() - start_total:.2f} 秒")