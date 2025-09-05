import numpy as np
from scipy.optimize import differential_evolution
import time
import multiprocessing as mp
from functools import partial
from joblib import Parallel, delayed  # 更好的并行库

# Constants from 1.py
g = 9.80665
epsilon = 1e-12

# Problem parameters (保持不变)
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

def calc_trajectory_points(angle, speed, drop_delay, det_delay):
    """Calculate drop and detonation points"""
    uav_dir = np.array([np.cos(angle), np.sin(angle), 0.0])
    drop_point = fy1_param["init_pos"] + uav_dir * speed * drop_delay
    
    horizontal_dist = speed * det_delay
    det_xy = drop_point[:2] + uav_dir[:2] * horizontal_dist
    det_z = drop_point[2] - 0.5 * g * det_delay ** 2
    det_point = np.array([det_xy[0], det_xy[1], det_z])
    
    return drop_point, det_point

def generate_target_samples(target, density="high"):
    """Generate target sampling points"""
    center = target["center"]
    r = target["r"] 
    h = target["h"]
    
    if density == "low":
        num_circle, num_height = 8, 4
    elif density == "medium":
        num_circle, num_height = 16, 8
    else:  # high
        num_circle, num_height = 24, 12
    
    samples = []
    angles = np.linspace(0, 2*np.pi, num_circle, endpoint=False)
    heights = np.linspace(center[2], center[2] + h, num_height)
    
    # Surface sampling
    for z in heights:
        for angle in angles:
            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)
            samples.append([x, y, z])
    
    # Add center axis points
    for z in heights:
        samples.append([center[0], center[1], z])
    
    return np.array(samples)

def line_sphere_intersect(start, end, sphere_center, radius):
    """Enhanced line-sphere intersection test"""
    d = end - start
    f = start - sphere_center
    
    a = np.dot(d, d)
    if a < epsilon:
        return np.linalg.norm(f) <= radius * 1.001
    
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - (radius * 1.001) ** 2
    
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return False
    
    sqrt_d = np.sqrt(discriminant)
    t1 = (-b - sqrt_d) / (2 * a)
    t2 = (-b + sqrt_d) / (2 * a)
    
    return (t1 <= 1.001 and t1 >= -0.001) or (t2 <= 1.001 and t2 >= -0.001) or (t1 < -0.001 and t2 > 1.001)

def evaluate_single_timepoint(t, missile_dir, t_det, det_point, target_samples):
    """评估单个时间点的遮蔽情况（用于并行化）"""
    missile_pos = missile_m1["init_pos"] + missile_dir * missile_m1["speed"] * t
    
    sink_time = t - t_det
    smoke_center = np.array([
        det_point[0],
        det_point[1], 
        det_point[2] - smoke_param["sink_speed"] * sink_time
    ])
    
    # Check if all target points are shielded
    for target_point in target_samples:
        if not line_sphere_intersect(missile_pos, target_point, smoke_center, smoke_param["r"]):
            return False
    return True

def evaluate_shielding_time_parallel(params, time_resolution=2000, n_jobs=-1):
    """并行化的遮蔽时间评估函数"""
    angle, speed, drop_delay, det_delay = params
    
    # Parameter validation
    if not (70 <= speed <= 140 and 0.05 <= drop_delay <= 10 and 0.05 <= det_delay <= 10):
        return 0
    
    try:
        drop_point, det_point = calc_trajectory_points(angle, speed, drop_delay, det_delay)
        
        # Missile trajectory
        missile_dir = (fake_target - missile_m1["init_pos"]) / np.linalg.norm(fake_target - missile_m1["init_pos"])
        
        # Time window
        t_det = drop_delay + det_delay
        t_end = t_det + smoke_param["valid_time"]
        
        # Time sampling
        time_samples = np.linspace(t_det, t_end, time_resolution)
        target_samples = generate_target_samples(real_target, "medium")
        dt = (t_end - t_det) / (time_resolution - 1) if time_resolution > 1 else 0
        
        # 并行评估每个时间点
        if n_jobs == -1:
            n_jobs = min(mp.cpu_count(), 56)  # 限制并行数避免过载
        
        results = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(evaluate_single_timepoint)(t, missile_dir, t_det, det_point, target_samples) 
            for t in time_samples
        )
        
        # 计算有效时间
        valid_time = sum(results) * dt
        return valid_time
        
    except:
        return 0

def objective_function_parallel(params):
    """并行化的目标函数"""
    return -evaluate_shielding_time_parallel(params, time_resolution=3000)

def ultra_parallel_grid_search(angle_range, speed_range, drop_range, det_range, n_jobs=-1):
    """超并行网格搜索"""
    print(f"Starting ultra-parallel grid search...")
    
    # Create parameter combinations
    angles = np.linspace(angle_range[0], angle_range[1], 25)
    speeds = np.linspace(speed_range[0], speed_range[1], 20)
    drop_delays = np.linspace(drop_range[0], drop_range[1], 25)
    det_delays = np.linspace(det_range[0], det_range[1], 25)
    
    param_combinations = []
    for angle in angles:
        for speed in speeds:
            for drop_delay in drop_delays:
                for det_delay in det_delays:
                    param_combinations.append([angle, speed, drop_delay, det_delay])
    
    print(f"Evaluating {len(param_combinations)} parameter combinations with ultra-parallelization...")
    
    # 超并行评估 - 使用所有可用核心
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    start_time = time.time()
    results = Parallel(n_jobs=n_jobs, backend='multiprocessing', verbose=1)(
        delayed(evaluate_shielding_time_parallel)(params, 2000, 1) for params in param_combinations
    )
    
    print(f"Grid search completed in {time.time() - start_time:.2f} seconds")
    
    # Find best result
    best_idx = np.argmax(results)
    best_params = param_combinations[best_idx]
    best_score = results[best_idx]
    
    return best_params, best_score

def adaptive_optimization_enhanced():
    """增强的自适应多阶段优化"""
    print("=== Enhanced Adaptive Multi-Stage Optimization ===")
    print(f"Using {mp.cpu_count()} CPU cores for maximum parallelization")
    
    # Stage 1: 超并行网格搜索
    print("\nStage 1: Ultra-parallel grid search...")
    stage1_params, stage1_score = ultra_parallel_grid_search(
        angle_range=(2.8, 3.4),  # 基于之前结果聚焦170-190度
        speed_range=(70, 100),
        drop_range=(0.05, 2.0),
        det_range=(0.05, 3.0),
        n_jobs=-1  # 使用所有核心
    )
    
    print(f"Stage 1 best: {stage1_score:.6f}s at angle={np.degrees(stage1_params[0]):.2f}°")
    
    # Stage 2: 并行差分进化
    print("\nStage 2: Parallel differential evolution...")
    
    margin = [0.15, 8, 0.4, 0.4]
    bounds = [
        (max(0, stage1_params[0] - margin[0]), min(2*np.pi, stage1_params[0] + margin[0])),
        (max(70, stage1_params[1] - margin[1]), min(140, stage1_params[1] + margin[1])),
        (max(0.05, stage1_params[2] - margin[2]), min(10, stage1_params[2] + margin[2])),
        (max(0.05, stage1_params[3] - margin[3]), min(10, stage1_params[3] + margin[3]))
    ]
    
    result = differential_evolution(
        objective_function_parallel,
        bounds,
        popsize=25,
        maxiter=80,
        disp=True,
        workers=min(mp.cpu_count()//2, 56),  # 使用一半核心避免过载
        seed=42,
        x0=stage1_params,
        atol=1e-7,
        tol=1e-7
    )
    
    stage2_params = result.x
    stage2_score = -result.fun
    
    print(f"Stage 2 best: {stage2_score:.6f}s")
    
    # Stage 3: 超高精度并行评估
    print("\nStage 3: Ultra-precise parallel evaluation...")
    final_score = evaluate_shielding_time_parallel(stage2_params, time_resolution=8000, n_jobs=-1)
    
    return stage2_params, final_score

def detailed_parallel_evaluation(params):
    """并行化的详细评估"""
    angle, speed, drop_delay, det_delay = params
    
    print(f"Detailed parallel evaluation: angle={np.degrees(angle):.3f}°, speed={speed:.3f}, "
          f"drop={drop_delay:.4f}s, det={det_delay:.4f}s")
    
    drop_point, det_point = calc_trajectory_points(angle, speed, drop_delay, det_delay)
    missile_dir = (fake_target - missile_m1["init_pos"]) / np.linalg.norm(fake_target - missile_m1["init_pos"])
    
    t_det = drop_delay + det_delay
    t_end = t_det + smoke_param["valid_time"]
    
    # 超高分辨率时间采样
    dt = 0.0005
    t_list = np.arange(t_det, t_end + dt, dt)
    target_samples = generate_target_samples(real_target, "high")
    
    print(f"Evaluating {len(t_list)} time points with {len(target_samples)} target samples...")
    
    # 并行评估所有时间点
    results = Parallel(n_jobs=-1, backend='threading', verbose=1)(
        delayed(evaluate_single_timepoint)(t, missile_dir, t_det, det_point, target_samples) 
        for t in t_list
    )
    
    # 分析结果
    valid_total = sum(results) * dt
    shield_segments = []
    prev_valid = False
    
    for i, (t, current_valid) in enumerate(zip(t_list, results)):
        if current_valid and not prev_valid:
            shield_segments.append({"start": t})
        elif not current_valid and prev_valid and shield_segments:
            shield_segments[-1]["end"] = t - dt
        prev_valid = current_valid
    
    # Handle final segment
    if shield_segments and "end" not in shield_segments[-1]:
        shield_segments[-1]["end"] = t_end
    
    return valid_total, drop_point, det_point, shield_segments

if __name__ == "__main__":
    start_time = time.time()
    print(f"Starting ULTRA-PARALLEL optimization with {mp.cpu_count()} CPU cores")
    
    # 运行增强的并行优化
    opt_params, opt_score = adaptive_optimization_enhanced()
    
    print(f"\nOptimization completed in {time.time() - start_time:.2f} seconds")
    print(f"Optimized shielding time: {opt_score:.8f} seconds")
    
    # 并行详细评估
    print("\nPerforming detailed parallel evaluation...")
    final_time, drop_pt, det_pt, segments = detailed_parallel_evaluation(opt_params)
    
    # Results summary
    print("\n" + "="*80)
    print("ULTRA-PARALLEL OPTIMIZATION RESULTS")
    print("="*80)
    print(f"Optimal flight angle: {np.degrees(opt_params[0]):.6f}°")
    print(f"Optimal flight speed: {opt_params[1]:.6f} m/s")
    print(f"Optimal drop delay: {opt_params[2]:.6f} s")
    print(f"Optimal detonation delay: {opt_params[3]:.6f} s")
    print(f"Drop point: [{drop_pt[0]:.6f}, {drop_pt[1]:.6f}, {drop_pt[2]:.6f}]")
    print(f"Detonation point: [{det_pt[0]:.6f}, {det_pt[1]:.6f}, {det_pt[2]:.6f}]")
    print(f"Maximum shielding time: {final_time:.8f} seconds")
    print("="*80)
    
    # Segment details
    if segments:
        print("\nShielding time segments:")
        total_duration = 0
        for i, seg in enumerate(segments, 1):
            duration = seg["end"] - seg["start"]
            total_duration += duration
            print(f"Segment {i}: {seg['start']:.6f}s ~ {seg['end']:.6f}s (duration: {duration:.6f}s)")
        print(f"Total verified shielding time: {total_duration:.8f}s")
        
        if final_time >= 4.75:
            print(f"\n🎯 OUTSTANDING! {final_time:.8f}s - Extremely close to 4.8s!")
        elif final_time >= 4.6:
            print(f"\n📈 EXCELLENT! {final_time:.8f}s - Very close to 4.8s!")
    
    print(f"\nTotal computation time: {time.time() - start_time:.2f} seconds")
    print(f"FINAL ANSWER: Maximum shielding time = {final_time:.8f} seconds")