import numpy as np
from scipy.optimize import differential_evolution
import time
import multiprocessing as mp
from joblib import Parallel, delayed

# Constants from 1.py
g = 9.80665
epsilon = 1e-12

# Problem parameters
fake_target = np.array([0.0, 0.0, 0.0])
real_target = {
    "center": np.array([0.0, 200.0, 0.0]),
    "r": 7.0,
    "h": 10.0
}

smoke_param = {
    "r": 10.0,
    "sink_speed": 3.0,
    "valid_time": 20.0
}

missile_m1 = {
    "init_pos": np.array([20000.0, 0.0, 2000.0]),
    "speed": 300.0
}

fy1_param = {
    "init_pos": np.array([17800.0, 0.0, 1800.0]),
}

def calc_drop_point(uav_init_pos, uav_dir, uav_speed, drop_delay):
    """计算烟幕弹投放点（与2.py保持一致）"""
    flight_dist = uav_speed * drop_delay
    drop_point = uav_init_pos + uav_dir * flight_dist
    return drop_point

def calc_det_point(drop_point, uav_dir, uav_speed, det_delay, g):
    """计算烟幕弹起爆点（与2.py保持一致）"""
    horizontal_dist = uav_speed * det_delay
    det_xy = drop_point[:2] + uav_dir[:2] * horizontal_dist
    
    drop_h = 0.5 * g * det_delay ** 2
    det_z = drop_point[2] - drop_h
    
    return np.array([det_xy[0], det_xy[1], det_z])

def generate_medium_density_samples(target, num_circle=20, num_height=8):
    """生成中等密度目标采样点（与2.py类似精度）"""
    samples = []
    center = target["center"]
    r = target["r"]
    h = target["h"]
    center_xy = center[:2]
    min_z = center[2]
    max_z = center[2] + h
    
    # 外表面采样
    theta = np.linspace(0, 2*np.pi, num_circle, endpoint=False)
    heights = np.linspace(min_z, max_z, num_height, endpoint=True)
    
    # 底面和顶面
    for th in theta:
        x = center_xy[0] + r * np.cos(th)
        y = center_xy[1] + r * np.sin(th)
        samples.append([x, y, min_z])
        samples.append([x, y, max_z])
    
    # 侧面
    for z in heights:
        for th in theta:
            x = center_xy[0] + r * np.cos(th)
            y = center_xy[1] + r * np.sin(th)
            samples.append([x, y, z])
    
    # 中轴线关键点
    samples.extend([
        [center_xy[0], center_xy[1], min_z],
        [center_xy[0], center_xy[1], min_z + h/2],
        [center_xy[0], center_xy[1], max_z]
    ])
    
    return np.array(samples)

def is_segment_intersect_sphere(M, P, C, r):
    """线段-球相交判定（与2.py保持一致）"""
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

def is_target_shielded(missile_pos, smoke_center, smoke_r, target_samples):
    """判定目标是否被遮蔽（与2.py保持一致）"""
    for p in target_samples:
        if not is_segment_intersect_sphere(missile_pos, p, smoke_center, smoke_r):
            return False
    return True

def evaluate_shielding_moderate(params):
    """中等精度的遮蔽时间评估（类似2.py的精度）"""
    dir_angle, speed, drop_delay, det_delay = params
    
    # 参数验证
    if not (70 <= speed <= 140 and 0.1 <= drop_delay <= 10 and 0.1 <= det_delay <= 10):
        return 0
    
    try:
        # 将角度转换为方向向量
        dir_xy = np.array([np.cos(dir_angle), np.sin(dir_angle)])
        uav_dir = np.array([dir_xy[0], dir_xy[1], 0.0])
        
        # 计算投放点和起爆点
        drop_point = calc_drop_point(
            uav_init_pos=fy1_param["init_pos"],
            uav_dir=uav_dir,
            uav_speed=speed,
            drop_delay=drop_delay
        )
        
        det_point = calc_det_point(
            drop_point=drop_point,
            uav_dir=uav_dir,
            uav_speed=speed,
            det_delay=det_delay,
            g=g
        )
        
        # 导弹方向计算
        missile_vec = fake_target - missile_m1["init_pos"]
        missile_dir = missile_vec / np.linalg.norm(missile_vec)
        
        # 时间窗口
        t_det = drop_delay + det_delay
        t_start = t_det
        t_end = t_det + smoke_param["valid_time"]
        
        # 使用中等密度时间采样（类似2.py的精度）
        dt = 0.001  # 1毫秒精度
        t_list = np.arange(t_start, t_end + dt, dt)
        
        # 生成中等密度目标采样点
        target_samples = generate_medium_density_samples(real_target, num_circle=30, num_height=10)
        
        # 计算遮蔽时间
        valid_total = 0.0
        
        for t in t_list:
            missile_pos = missile_m1["init_pos"] + missile_dir * missile_m1["speed"] * t
            sink_time = t - t_det
            smoke_center = np.array([
                det_point[0],
                det_point[1],
                det_point[2] - smoke_param["sink_speed"] * sink_time
            ])
            
            if is_target_shielded(missile_pos, smoke_center, smoke_param["r"], target_samples):
                valid_total += dt
        
        return valid_total
        
    except:
        return 0

def objective_function_moderate(params):
    """目标函数"""
    return -evaluate_shielding_moderate(params)

def parallel_grid_search_moderate(angle_range, speed_range, drop_range, det_range, n_jobs=-1):
    """中等精度的并行网格搜索"""
    print(f"Starting moderate precision parallel grid search...")
    
    # 创建参数组合（适中的网格密度）
    angles = np.linspace(angle_range[0], angle_range[1], 20)  # 降低到20个角度
    speeds = np.linspace(speed_range[0], speed_range[1], 15)  # 15个速度
    drop_delays = np.linspace(drop_range[0], drop_range[1], 20)  # 20个投放延迟
    det_delays = np.linspace(det_range[0], det_range[1], 20)   # 20个起爆延迟
    
    param_combinations = []
    for angle in angles:
        for speed in speeds:
            for drop_delay in drop_delays:
                for det_delay in det_delays:
                    param_combinations.append([angle, speed, drop_delay, det_delay])
    
    print(f"Evaluating {len(param_combinations)} parameter combinations...")
    
    # 并行评估
    if n_jobs == -1:
        n_jobs = min(mp.cpu_count(), 32)  # 限制核心数
    
    start_time = time.time()
    results = Parallel(n_jobs=n_jobs, backend='multiprocessing', verbose=1)(
        delayed(evaluate_shielding_moderate)(params) for params in param_combinations
    )
    
    print(f"Grid search completed in {time.time() - start_time:.2f} seconds")
    
    # 找到最佳结果
    best_idx = np.argmax(results)
    best_params = param_combinations[best_idx]
    best_score = results[best_idx]
    
    return best_params, best_score

def moderate_optimization():
    """中等精度的多阶段优化"""
    print("=== Moderate Precision Multi-Stage Optimization ===")
    print(f"Using {mp.cpu_count()} CPU cores with moderate precision")
    
    # 第一阶段：并行网格搜索
    print("\nStage 1: Parallel grid search...")
    stage1_params, stage1_score = parallel_grid_search_moderate(
        angle_range=(2.8, 3.4),  # 基于之前结果聚焦170-190度
        speed_range=(70, 120),
        drop_range=(0.1, 3.0),
        det_range=(0.5, 5.0),
        n_jobs=-1
    )
    
    print(f"Stage 1 best: {stage1_score:.4f}s at angle={np.degrees(stage1_params[0]):.2f}°")
    
    # 第二阶段：差分进化优化
    print("\nStage 2: Differential evolution...")
    
    margin = [0.2, 10, 0.5, 0.5]
    bounds = [
        (max(0, stage1_params[0] - margin[0]), min(2*np.pi, stage1_params[0] + margin[0])),
        (max(70, stage1_params[1] - margin[1]), min(140, stage1_params[1] + margin[1])),
        (max(0.1, stage1_params[2] - margin[2]), min(10, stage1_params[2] + margin[2])),
        (max(0.1, stage1_params[3] - margin[3]), min(10, stage1_params[3] + margin[3]))
    ]
    
    result = differential_evolution(
        objective_function_moderate,
        bounds,
        popsize=20,  # 适中的种群大小
        maxiter=50,  # 适中的迭代次数
        disp=True,
        workers=min(mp.cpu_count()//2, 16),  # 适中的并行度
        seed=42,
        x0=stage1_params,
        atol=1e-6,
        tol=1e-6
    )
    
    stage2_params = result.x
    stage2_score = -result.fun
    
    print(f"Stage 2 best: {stage2_score:.6f}s")
    
    return stage2_params, stage2_score

def detailed_evaluation_moderate(params):
    """中等精度的详细评估"""
    angle, speed, drop_delay, det_delay = params
    
    print(f"Detailed evaluation: angle={np.degrees(angle):.3f}°, speed={speed:.3f}, "
          f"drop={drop_delay:.4f}s, det={det_delay:.4f}s")
    
    # 将角度转换为方向向量
    dir_xy = np.array([np.cos(angle), np.sin(angle)])
    uav_dir = np.array([dir_xy[0], dir_xy[1], 0.0])
    
    drop_point = calc_drop_point(fy1_param["init_pos"], uav_dir, speed, drop_delay)
    det_point = calc_det_point(drop_point, uav_dir, speed, det_delay, g)
    missile_dir = (fake_target - missile_m1["init_pos"]) / np.linalg.norm(fake_target - missile_m1["init_pos"])
    
    t_det = drop_delay + det_delay
    t_end = t_det + smoke_param["valid_time"]
    
    # 精细时间采样（类似2.py的最终验证）
    fine_dt = 0.0001
    t_list = np.arange(t_det, t_end + fine_dt, fine_dt)
    target_samples = generate_medium_density_samples(real_target, num_circle=40, num_height=15)
    
    print(f"Evaluating {len(t_list)} time points with {len(target_samples)} target samples...")
    
    valid_total = 0.0
    shield_segments = []
    prev_valid = False
    
    for i, t in enumerate(t_list):
        if i % 50000 == 0:
            progress = 100 * i / len(t_list)
            print(f"Progress: {progress:.1f}%")
        
        missile_pos = missile_m1["init_pos"] + missile_dir * missile_m1["speed"] * t
        
        sink_time = t - t_det
        smoke_center = np.array([
            det_point[0],
            det_point[1],
            det_point[2] - smoke_param["sink_speed"] * sink_time
        ])
        
        current_valid = is_target_shielded(missile_pos, smoke_center, smoke_param["r"], target_samples)
        
        if current_valid:
            valid_total += fine_dt
        
        # 记录遮蔽时间段
        if current_valid and not prev_valid:
            shield_segments.append({"start": t})
        elif not current_valid and prev_valid and shield_segments:
            shield_segments[-1]["end"] = t - fine_dt
        
        prev_valid = current_valid
    
    # 处理最后一个遮蔽段
    if shield_segments and "end" not in shield_segments[-1]:
        shield_segments[-1]["end"] = t_end
    
    return valid_total, drop_point, det_point, shield_segments

if __name__ == "__main__":
    start_time = time.time()
    print(f"Starting MODERATE PRECISION optimization with {mp.cpu_count()} CPU cores")
    
    # 运行中等精度优化
    opt_params, opt_score = moderate_optimization()
    
    print(f"\nOptimization completed in {time.time() - start_time:.2f} seconds")
    print(f"Optimized shielding time: {opt_score:.6f} seconds")
    
    # 详细评估
    print("\nPerforming detailed evaluation...")
    final_time, drop_pt, det_pt, segments = detailed_evaluation_moderate(opt_params)
    
    # 结果汇总
    print("\n" + "="*70)
    print("MODERATE PRECISION OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Optimal flight angle: {np.degrees(opt_params[0]):.4f}°")
    print(f"Optimal flight speed: {opt_params[1]:.4f} m/s")
    print(f"Optimal drop delay: {opt_params[2]:.4f} s")
    print(f"Optimal detonation delay: {opt_params[3]:.4f} s")
    print(f"Drop point: [{drop_pt[0]:.4f}, {drop_pt[1]:.4f}, {drop_pt[2]:.4f}]")
    print(f"Detonation point: [{det_pt[0]:.4f}, {det_pt[1]:.4f}, {det_pt[2]:.4f}]")
    print(f"Maximum shielding time: {final_time:.6f} seconds")
    print("="*70)
    
    # 遮蔽时间段详情
    if segments:
        print("\nShielding time segments:")
        total_duration = 0
        for i, seg in enumerate(segments, 1):
            duration = seg["end"] - seg["start"]
            total_duration += duration
            print(f"Segment {i}: {seg['start']:.4f}s ~ {seg['end']:.4f}s (duration: {duration:.4f}s)")
        print(f"Total verified shielding time: {total_duration:.6f}s")
        
        if final_time >= 4.5:
            print(f"\n🎯 EXCELLENT! {final_time:.6f}s - Great result!")
        elif final_time >= 4.0:
            print(f"\n📈 GOOD! {final_time:.6f}s - Solid performance!")
    
    print(f"\nTotal computation time: {time.time() - start_time:.2f} seconds")
    print(f"FINAL ANSWER: Maximum shielding time = {final_time:.6f} seconds")