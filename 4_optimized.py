import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import joblib
from numba import jit, prange
import time

# 检测GPU可用性
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print(f"检测到GPU: {cp.cuda.runtime.getDeviceCount()}张")
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False
    print("GPU不可用，使用CPU计算")

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
@jit(nopython=True)
def calc_uav_direction_numba(angle_deg):
    """Numba加速的无人机飞行方向计算"""
    angle_rad = np.radians(angle_deg)
    return np.array([np.cos(angle_rad), np.sin(angle_rad)])

@jit(nopython=True)
def calc_drop_point_numba(uav_init_pos, uav_speed, drop_delay, angle_deg):
    """Numba加速的投放点计算"""
    dir_vec_xy = calc_uav_direction_numba(angle_deg)
    flight_dist = uav_speed * drop_delay
    drop_xy = uav_init_pos[:2] + dir_vec_xy * flight_dist
    return np.array([drop_xy[0], drop_xy[1], uav_init_pos[2]])

@jit(nopython=True)
def calc_det_point_numba(drop_point, uav_speed, det_delay, angle_deg):
    """Numba加速的起爆点计算"""
    dir_vec_xy = calc_uav_direction_numba(angle_deg)
    horizontal_dist = uav_speed * det_delay
    det_xy = drop_point[:2] + dir_vec_xy * horizontal_dist
    drop_h = 0.5 * g * det_delay ** 2
    det_z = drop_point[2] - drop_h
    return np.array([det_xy[0], det_xy[1], det_z])

@jit(nopython=True)
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

@jit(nopython=True)
def check_all_samples_shielded_numba(missile_pos, smoke_center, smoke_r, target_samples):
    """Numba加速的批量遮蔽检测"""
    for i in range(len(target_samples)):
        if not is_segment_intersect_sphere_numba(missile_pos, target_samples[i], smoke_center, smoke_r):
            return False
    return True

@jit(nopython=True)
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

# -------------------------- 4. GPU加速版本 --------------------------
def calc_shield_time_gpu_accelerated(params, uav_names, target_samples):
    """GPU加速版本的遮蔽时间计算"""
    if not GPU_AVAILABLE or len(target_samples) < 1000:
        return calc_total_shield_time_parallel(params, uav_names, target_samples)
    
    try:
        # 将数据移到GPU
        target_samples_gpu = cp.asarray(target_samples)
        
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
        
        # 计算每架无人机的起爆点和起爆时间
        uav_data = {}
        for uav_name in uav_names:
            uav_pos = uav_positions[uav_name]
            param = uav_params[uav_name]
            
            drop_point = calc_drop_point_numba(uav_pos, param["speed"], param["drop_delay"], param["angle"])
            det_point = calc_det_point_numba(drop_point, param["speed"], param["det_delay"], param["angle"])
            det_time = param["drop_delay"] + param["det_delay"]
            
            uav_data[uav_name] = {
                "det_point": det_point,
                "det_time": det_time,
                "param": param
            }
        
        # 计算导弹飞行方向
        missile_vec = fake_target - missile_m1["init_pos"]
        missile_dist = np.linalg.norm(missile_vec)
        if missile_dist < epsilon:
            missile_dir = np.array([0.0, 0.0, 0.0])
        else:
            missile_dir = missile_vec / missile_dist
        
        # 确定时间范围
        min_det_time = min(data["det_time"] for data in uav_data.values())
        max_det_time = max(data["det_time"] for data in uav_data.values())
        
        t_start = min_det_time
        t_end = max_det_time + smoke_param["valid_time"]
        t_list = cp.arange(t_start, t_end + dt, dt)
        
        # GPU批量计算
        valid_count = 0
        missile_init_pos_gpu = cp.asarray(missile_m1["init_pos"])
        missile_dir_gpu = cp.asarray(missile_dir)
        missile_speed = missile_m1["speed"]
        
        # 批量计算导弹轨迹
        missile_positions = missile_init_pos_gpu + missile_dir_gpu * missile_speed * t_list[:, cp.newaxis]
        
        for i, t in enumerate(cp.asnumpy(t_list)):
            missile_pos = cp.asnumpy(missile_positions[i])
            
            # 计算当前时刻的有效烟幕
            active_smoke_centers = []
            for uav_name, data in uav_data.items():
                det_time = data["det_time"]
                det_point = data["det_point"]
                
                if t >= det_time and t <= det_time + smoke_param["valid_time"]:
                    sink_time = t - det_time
                    smoke_center = np.array([
                        det_point[0],
                        det_point[1],
                        det_point[2] - smoke_param["sink_speed"] * sink_time
                    ])
                    active_smoke_centers.append(smoke_center)
            
            if active_smoke_centers:
                active_smoke_centers = np.array(active_smoke_centers)
                if check_any_smoke_shields_numba(missile_pos, active_smoke_centers, smoke_param["r"], target_samples):
                    valid_count += 1
        
        return valid_count * dt
        
    except Exception as e:
        print(f"GPU计算错误，回退到CPU: {e}")
        return calc_total_shield_time_parallel(params, uav_names, target_samples)

# -------------------------- 5. CPU多核并行版本 --------------------------
def calc_total_shield_time_parallel(params, uav_names, target_samples, n_jobs=None):
    """CPU多核并行版本的遮蔽时间计算"""
    if n_jobs is None:
        n_jobs = min(cpu_count(), 56)  # 使用一半核心，保留其他任务使用
    
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
        
        # 计算每架无人机的起爆点和起爆时间
        uav_data = {}
        for uav_name in uav_names:
            uav_pos = uav_positions[uav_name]
            param = uav_params[uav_name]
            
            drop_point = calc_drop_point_numba(uav_pos, param["speed"], param["drop_delay"], param["angle"])
            det_point = calc_det_point_numba(drop_point, param["speed"], param["det_delay"], param["angle"])
            det_time = param["drop_delay"] + param["det_delay"]
            
            uav_data[uav_name] = {
                "det_point": det_point,
                "det_time": det_time,
                "param": param
            }
        
        # 计算导弹飞行方向
        missile_vec = fake_target - missile_m1["init_pos"]
        missile_dist = np.linalg.norm(missile_vec)
        if missile_dist < epsilon:
            missile_dir = np.array([0.0, 0.0, 0.0])
        else:
            missile_dir = missile_vec / missile_dist
        
        # 确定时间范围
        min_det_time = min(data["det_time"] for data in uav_data.values())
        max_det_time = max(data["det_time"] for data in uav_data.values())
        
        t_start = min_det_time
        t_end = max_det_time + smoke_param["valid_time"]
        t_list = np.arange(t_start, t_end + dt, dt)
        
        # 并行计算时间块
        def calc_time_chunk(time_chunk):
            """计算时间片段的遮蔽状态"""
            valid_time = 0.0
            
            for t in time_chunk:
                # 计算导弹位置
                missile_pos = missile_m1["init_pos"] + missile_dir * missile_m1["speed"] * t
                
                # 计算所有有效烟幕的位置
                active_smoke_centers = []
                for uav_name, data in uav_data.items():
                    det_time = data["det_time"]
                    det_point = data["det_point"]
                    
                    # 检查烟幕是否已起爆且仍有效
                    if t >= det_time and t <= det_time + smoke_param["valid_time"]:
                        sink_time = t - det_time
                        smoke_center = np.array([
                            det_point[0],
                            det_point[1],
                            det_point[2] - smoke_param["sink_speed"] * sink_time
                        ])
                        active_smoke_centers.append(smoke_center)
                
                # 判断是否被遮蔽
                if active_smoke_centers:
                    active_smoke_centers = np.array(active_smoke_centers)
                    if check_any_smoke_shields_numba(missile_pos, active_smoke_centers, smoke_param["r"], target_samples):
                        valid_time += dt
            
            return valid_time
        
        # 将时间序列分割为多个块进行并行计算
        chunk_size = max(100, len(t_list) // (n_jobs * 4))  # 确保每个块有足够的工作量
        time_chunks = [t_list[i:i + chunk_size] for i in range(0, len(t_list), chunk_size)]
        
        # 使用joblib并行计算
        results = joblib.Parallel(n_jobs=n_jobs, backend='threading', prefer="threads")(
            joblib.delayed(calc_time_chunk)(chunk) for chunk in time_chunks
        )
        
        return sum(results)
        
    except Exception as e:
        print(f"并行计算错误: {e}")
        return 0.0

# -------------------------- 6. 标准版本 (兜底) --------------------------
def calc_total_shield_time_multi_uav(params, uav_names, target_samples):
    """标准版本的遮蔽时间计算"""
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
        
        # 计算每架无人机的起爆点和起爆时间
        uav_data = {}
        for uav_name in uav_names:
            uav_pos = uav_positions[uav_name]
            param = uav_params[uav_name]
            
            drop_point = calc_drop_point(
                uav_pos, param["speed"], param["drop_delay"], 
                fake_target, param["angle"]
            )
            
            det_point = calc_det_point(
                drop_point, param["speed"], param["det_delay"], g,
                fake_target, param["angle"], uav_pos
            )
            
            det_time = param["drop_delay"] + param["det_delay"]
            
            uav_data[uav_name] = {
                "det_point": det_point,
                "det_time": det_time,
                "param": param
            }
        
        # 计算导弹飞行方向
        missile_vec = fake_target - missile_m1["init_pos"]
        missile_dist = np.linalg.norm(missile_vec)
        if missile_dist < epsilon:
            missile_dir = np.array([0.0, 0.0, 0.0])
        else:
            missile_dir = missile_vec / missile_dist
        
        # 确定时间范围
        min_det_time = min(data["det_time"] for data in uav_data.values())
        max_det_time = max(data["det_time"] for data in uav_data.values())
        
        t_start = min_det_time
        t_end = max_det_time + smoke_param["valid_time"]
        t_list = np.arange(t_start, t_end + dt, dt)
        
        # 逐时刻计算遮蔽状态
        valid_total = 0.0
        
        for t in t_list:
            # 计算导弹位置
            missile_pos = missile_m1["init_pos"] + missile_dir * missile_m1["speed"] * t
            
            # 计算所有有效烟幕的位置
            active_smoke_centers = []
            for uav_name, data in uav_data.items():
                det_time = data["det_time"]
                det_point = data["det_point"]
                
                # 检查烟幕是否已起爆且仍有效
                if t >= det_time and t <= det_time + smoke_param["valid_time"]:
                    sink_time = t - det_time
                    smoke_center = np.array([
                        det_point[0],
                        det_point[1],
                        det_point[2] - smoke_param["sink_speed"] * sink_time
                    ])
                    active_smoke_centers.append(smoke_center)
            
            # 判断是否被遮蔽
            if active_smoke_centers:
                for smoke_center in active_smoke_centers:
                    if check_all_samples_shielded_numba(missile_pos, smoke_center, smoke_param["r"], target_samples):
                        valid_total += dt
                        break
        
        return valid_total
        
    except Exception as e:
        print(f"计算错误: {e}")
        return 0.0

# -------------------------- 7. 智能优化策略 --------------------------
def ultimate_optimization_strategy(uav_names, target_samples):
    """终极优化策略：自动选择最佳计算方式"""
    
    print("=== 服务器资源检测 ===")
    print(f"CPU核心数: {cpu_count()}")
    
    if GPU_AVAILABLE:
        print(f"GPU数量: {cp.cuda.runtime.getDeviceCount()}")
        try:
            for i in range(min(4, cp.cuda.runtime.getDeviceCount())):  # 只检测前4个GPU
                cp.cuda.Device(i).use()
                meminfo = cp.cuda.runtime.memGetInfo()
                print(f"GPU {i}: {meminfo[1]//1024**3:.1f}GB total, {meminfo[0]//1024**3:.1f}GB free")
        except:
            print("GPU信息获取失败")
    
    n_samples = len(target_samples)
    n_params = len(uav_names) * 4
    
    # 智能选择计算策略
    if GPU_AVAILABLE and n_samples > 2000:
        print("策略: GPU加速 + CPU并行")
        calc_func = calc_shield_time_gpu_accelerated
        n_workers = min(56, cpu_count() // 2)
    elif cpu_count() >= 50:
        print("策略: 高度CPU并行")
        calc_func = calc_total_shield_time_parallel
        n_workers = min(80, int(cpu_count() * 0.7))
    elif cpu_count() >= 8:
        print("策略: 标准并行")
        calc_func = calc_total_shield_time_parallel
        n_workers = min(cpu_count() - 2, 16)
    else:
        print("策略: 标准计算")
        calc_func = calc_total_shield_time_multi_uav
        n_workers = 1
    
    # 参数边界
    bounds = [(70, 140), (0.1, 5.0), (0.1, 5.0), (0, 360)] * len(uav_names)
    
    def objective(params):
        return -calc_func(params, uav_names, target_samples)
    
    print(f"\n=== 开始多阶段优化 ===")
    print(f"优化参数数量: {len(bounds)}")
    print(f"目标采样点数: {n_samples}")
    
    start_time = time.time()
    
    # 阶段1: 快速全局搜索
    print("\n阶段1: 快速全局搜索 (30%的计算量)...")
    result1 = differential_evolution(
        objective,
        bounds,
        maxiter=50,
        popsize=15,
        workers=max(1, n_workers//2) if n_workers > 1 else 1,
        seed=42,
        disp=True
    )
    
    stage1_time = time.time() - start_time
    print(f"阶段1完成，用时: {stage1_time:.2f}秒，最优值: {-result1.fun:.6f}")
    
    # 阶段2: 精细搜索
    print("\n阶段2: 基于最优结果的精细搜索 (70%的计算量)...")
    
    # 缩小搜索范围到最优解附近
    best_params = result1.x
    refined_bounds = []
    for i, (low, high) in enumerate(bounds):
        center = best_params[i]
        range_size = (high - low) * 0.15  # 缩小到15%范围
        new_low = max(low, center - range_size/2)
        new_high = min(high, center + range_size/2)
        refined_bounds.append((new_low, new_high))
    
    result2 = differential_evolution(
        objective,
        refined_bounds,
        maxiter=100,
        popsize=25,
        workers=n_workers if n_workers > 1 else 1,
        seed=43,
        polish=True,
        disp=True
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
    print(f"平均每次评估时间: {total_time/(result1.nfev + result2.nfev):.4f}秒")
    
    return final_result

def optimize_multi_uav(uav_names, target_samples):
    """优化多架无人机参数 - 主要接口函数"""
    return ultimate_optimization_strategy(uav_names, target_samples)

# -------------------------- 8. 主程序 --------------------------
if __name__ == "__main__":
    print("=== 多无人机协同烟幕干扰优化系统 (高性能版本) ===")
    
    # 生成目标采样点
    print("\n生成真目标采样点...")
    target_samples = generate_high_density_samples(real_target, num_circle=60, num_height=20)
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
            
            # 计算单架无人机的贡献时间（用于参考）
            single_shield_time = calc_total_shield_time_multi_uav(
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
            df.to_csv("result2.csv", index=False)
            print("已保存为CSV格式: result2.csv")
        
        print(f"\n*** 协同总遮蔽时间：{-result.fun:.6f} 秒 ***")
        
    else:
        print("优化失败:", result.message)
        print("尝试降低计算精度或减少优化参数...")

    print(f"\n程序执行完毕，总用时: {time.time() - start_total:.2f} 秒")