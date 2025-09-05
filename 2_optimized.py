import numpy as np
from scipy.optimize import differential_evolution
import time
import multiprocessing as mp
from functools import partial

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
    """Line-sphere intersection test"""
    d = end - start
    f = start - sphere_center
    
    a = np.dot(d, d)
    if a < epsilon:
        return np.linalg.norm(f) <= radius
    
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius * radius
    
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return False
    
    sqrt_d = np.sqrt(discriminant)
    t1 = (-b - sqrt_d) / (2 * a)
    t2 = (-b + sqrt_d) / (2 * a)
    
    return (t1 <= 1.0 and t1 >= 0.0) or (t2 <= 1.0 and t2 >= 0.0) or (t1 < 0.0 and t2 > 1.0)

def evaluate_shielding_time(params, time_resolution=1000):
    """Evaluate shielding time for given parameters"""
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
        
        valid_time = 0.0
        dt = (t_end - t_det) / (time_resolution - 1)
        
        for t in time_samples:
            missile_pos = missile_m1["init_pos"] + missile_dir * missile_m1["speed"] * t
            
            sink_time = t - t_det
            smoke_center = np.array([
                det_point[0],
                det_point[1], 
                det_point[2] - smoke_param["sink_speed"] * sink_time
            ])
            
            # Check if all target points are shielded
            all_shielded = True
            for target_point in target_samples:
                if not line_sphere_intersect(missile_pos, target_point, smoke_center, smoke_param["r"]):
                    all_shielded = False
                    break
            
            if all_shielded:
                valid_time += dt
        
        return valid_time
        
    except:
        return 0

def objective_function(params):
    """Objective function for optimization (negative for minimization)"""
    return -evaluate_shielding_time(params)

def parallel_grid_search(angle_range, speed_range, drop_range, det_range, num_workers=None):
    """Parallel grid search using multiprocessing"""
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 64)  # Use up to 64 cores
    
    print(f"Starting parallel grid search with {num_workers} workers...")
    
    # Create parameter combinations
    angles = np.linspace(angle_range[0], angle_range[1], 20)
    speeds = np.linspace(speed_range[0], speed_range[1], 15)
    drop_delays = np.linspace(drop_range[0], drop_range[1], 20)
    det_delays = np.linspace(det_range[0], det_range[1], 20)
    
    param_combinations = []
    for angle in angles:
        for speed in speeds:
            for drop_delay in drop_delays:
                for det_delay in det_delays:
                    param_combinations.append([angle, speed, drop_delay, det_delay])
    
    print(f"Evaluating {len(param_combinations)} parameter combinations...")
    
    # Parallel evaluation
    with mp.Pool(num_workers) as pool:
        results = pool.map(evaluate_shielding_time, param_combinations)
    
    # Find best result
    best_idx = np.argmax(results)
    best_params = param_combinations[best_idx]
    best_score = results[best_idx]
    
    return best_params, best_score

def adaptive_optimization():
    """Multi-stage adaptive optimization"""
    print("=== Adaptive Multi-Stage Optimization ===")
    
    # Stage 1: Coarse grid search around promising regions
    print("\nStage 1: Coarse parallel grid search...")
    stage1_params, stage1_score = parallel_grid_search(
        angle_range=(2.8, 3.4),  # Around 170-190 degrees based on previous results
        speed_range=(70, 100),
        drop_range=(0.05, 2.0),
        det_range=(0.05, 3.0),
        num_workers=56  # Use half your cores for this stage
    )
    
    print(f"Stage 1 best: {stage1_score:.4f}s at angle={np.degrees(stage1_params[0]):.1f}°")
    
    # Stage 2: Fine-tuned differential evolution
    print("\nStage 2: Fine-tuned differential evolution...")
    
    # Set bounds around stage 1 result
    margin = [0.2, 10, 0.5, 0.5]  # Margins for angle, speed, drop_delay, det_delay
    bounds = [
        (max(0, stage1_params[0] - margin[0]), min(2*np.pi, stage1_params[0] + margin[0])),
        (max(70, stage1_params[1] - margin[1]), min(140, stage1_params[1] + margin[1])),
        (max(0.05, stage1_params[2] - margin[2]), min(10, stage1_params[2] + margin[2])),
        (max(0.05, stage1_params[3] - margin[3]), min(10, stage1_params[3] + margin[3]))
    ]
    
    result = differential_evolution(
        objective_function,
        bounds,
        popsize=20,
        maxiter=100,
        disp=True,
        workers=56,  # Use remaining cores
        seed=42,
        x0=stage1_params,
        atol=1e-6,
        tol=1e-6
    )
    
    stage2_params = result.x
    stage2_score = -result.fun
    
    print(f"Stage 2 best: {stage2_score:.4f}s")
    
    # Stage 3: Ultra-precise evaluation
    print("\nStage 3: Ultra-precise final evaluation...")
    final_score = evaluate_shielding_time(stage2_params, time_resolution=5000)
    
    return stage2_params, final_score

def detailed_evaluation(params):
    """Detailed evaluation with segment analysis"""
    angle, speed, drop_delay, det_delay = params
    
    print(f"Detailed evaluation: angle={np.degrees(angle):.2f}°, speed={speed:.2f}, "
          f"drop={drop_delay:.3f}s, det={det_delay:.3f}s")
    
    drop_point, det_point = calc_trajectory_points(angle, speed, drop_delay, det_delay)
    missile_dir = (fake_target - missile_m1["init_pos"]) / np.linalg.norm(fake_target - missile_m1["init_pos"])
    
    t_det = drop_delay + det_delay
    t_end = t_det + smoke_param["valid_time"]
    
    # High-resolution time sampling
    dt = 0.001
    t_list = np.arange(t_det, t_end + dt, dt)
    target_samples = generate_target_samples(real_target, "high")
    
    valid_total = 0.0
    shield_segments = []
    prev_valid = False
    
    for t in t_list:
        missile_pos = missile_m1["init_pos"] + missile_dir * missile_m1["speed"] * t
        
        sink_time = t - t_det
        smoke_center = np.array([
            det_point[0],
            det_point[1],
            det_point[2] - smoke_param["sink_speed"] * sink_time
        ])
        
        current_valid = True
        for target_point in target_samples:
            if not line_sphere_intersect(missile_pos, target_point, smoke_center, smoke_param["r"]):
                current_valid = False
                break
        
        if current_valid:
            valid_total += dt
        
        # Track segments
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
    print(f"Using {mp.cpu_count()} CPU cores for optimization")
    
    # Run adaptive optimization
    opt_params, opt_score = adaptive_optimization()
    
    print(f"\nOptimization completed in {time.time() - start_time:.2f} seconds")
    print(f"Optimized shielding time: {opt_score:.6f} seconds")
    
    # Detailed final evaluation
    print("\nPerforming detailed final evaluation...")
    final_time, drop_pt, det_pt, segments = detailed_evaluation(opt_params)
    
    # Results summary
    print("\n" + "="*70)
    print("FINAL OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Optimal flight angle: {np.degrees(opt_params[0]):.3f}°")
    print(f"Optimal flight speed: {opt_params[1]:.3f} m/s")
    print(f"Optimal drop delay: {opt_params[2]:.3f} s")
    print(f"Optimal detonation delay: {opt_params[3]:.3f} s")
    print(f"Drop point: [{drop_pt[0]:.3f}, {drop_pt[1]:.3f}, {drop_pt[2]:.3f}]")
    print(f"Detonation point: [{det_pt[0]:.3f}, {det_pt[1]:.3f}, {det_pt[2]:.3f}]")
    print(f"Maximum shielding time: {final_time:.6f} seconds")
    print("="*70)
    
    # Segment details
    if segments:
        print("\nShielding time segments:")
        total_duration = 0
        for i, seg in enumerate(segments, 1):
            duration = seg["end"] - seg["start"]
            total_duration += duration
            print(f"Segment {i}: {seg['start']:.4f}s ~ {seg['end']:.4f}s (duration: {duration:.4f}s)")
        print(f"Total verified shielding time: {total_duration:.6f}s")
    
    print(f"\nTotal computation time: {time.time() - start_time:.2f} seconds")