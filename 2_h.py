import numpy as np
import time
from itertools import product
from multiprocessing import Pool, cpu_count
import functools

# -------------------------- 1. 常量与初始参数定义 --------------------------
g = 9.80665  # 重力加速度 (m/s²)
epsilon = 1e-12  # 数值计算保护阈值

# 目标定义
fake_target = np.array([0.0, 0.0, 0.0])  # 假目标（原点，导弹/无人机指向目标）
# 真目标：半径7m、高10m的圆柱体，底面圆心(0,200,0)
real_target = {
    "center": np.array([0.0, 200.0, 0.0]),  # 底面圆心
    "r": 7.0,  # 圆柱半径
    "h": 10.0   # 圆柱高度
}

# 无人机FY1参数
fy1_param = {
    "init_pos": np.array([17800.0, 0.0, 1800.0]),  # 初始位置
    "speed_range": [70.0, 140.0],  # 飞行速度范围(m/s)
    "drop_delay_range": [0.5, 5.0],  # 受领任务到投放的时间范围(s)
    "det_delay_range": [1.0, 8.0]    # 投放至起爆的时间范围(s)
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

dt = 0.01  # 时间步长（适度精度以提高搜索效率）


# -------------------------- 2. 核心位置计算函数 --------------------------
def calc_drop_point(uav_init_pos, uav_speed, drop_delay, flight_direction):
    """计算烟幕弹投放点（无人机等高度飞行，指定方向）"""
    flight_dist = uav_speed * drop_delay
    drop_xy = uav_init_pos[:2] + flight_direction * flight_dist
    drop_z = uav_init_pos[2]  # 等高度飞行
    
    return np.array([drop_xy[0], drop_xy[1], drop_z])


def calc_det_point(drop_point, uav_speed, det_delay, g, flight_direction):
    """计算烟幕弹起爆点（投放后水平沿无人机方向，竖直自由落体）"""
    horizontal_dist = uav_speed * det_delay
    det_xy = drop_point[:2] + flight_direction * horizontal_dist
    
    # 竖直方向自由落体
    drop_h = 0.5 * g * det_delay ** 2
    det_z = drop_point[2] - drop_h
    
    return np.array([det_xy[0], det_xy[1], det_z])


# -------------------------- 3. 优化的目标采样点生成 --------------------------
def generate_target_samples(target, num_circle=30, num_height=10):
    """生成目标采样点（优化密度以提高搜索效率）"""
    samples = []
    center = target["center"]
    r = target["r"]
    h = target["h"]
    center_xy = center[:2]
    min_z = center[2]
    max_z = center[2] + h
    
    # 1. 外表面采样
    theta = np.linspace(0, 2*np.pi, num_circle, endpoint=False)
    heights = np.linspace(min_z, max_z, num_height, endpoint=True)
    
    for z in heights:
        for th in theta:
            x = center_xy[0] + r * np.cos(th)
            y = center_xy[1] + r * np.sin(th)
            samples.append([x, y, z])
    
    # 2. 关键内部点
    for z in [min_z, min_z + h/2, max_z]:
        for rad in [0, r/2, r]:
            for th in np.linspace(0, 2*np.pi, 8, endpoint=False):
                x = center_xy[0] + rad * np.cos(th)
                y = center_xy[1] + rad * np.sin(th)
                samples.append([x, y, z])
    
    return np.unique(np.array(samples), axis=0)


# -------------------------- 4. 高精度几何判定函数 --------------------------
def is_segment_intersect_sphere(M, P, C, r):
    """高精度判定线段MP与球C(r)是否相交"""
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
    """判定真目标是否被完全遮蔽"""
    for p in target_samples:
        if not is_segment_intersect_sphere(missile_pos, p, smoke_center, smoke_r):
            return False
    return True


# -------------------------- 5. 遮蔽时长计算函数 --------------------------
def calculate_shield_time(uav_speed, drop_delay, det_delay, flight_direction):
    """计算给定参数下的遮蔽时长"""
    try:
        # 计算投放点和起爆点
        drop_point = calc_drop_point(
            fy1_param["init_pos"], uav_speed, drop_delay, flight_direction
        )
        
        det_point = calc_det_point(
            drop_point, uav_speed, det_delay, g, flight_direction
        )
        
        # 生成目标采样点
        target_samples = generate_target_samples(real_target)
        
        # 导弹飞行方向
        missile_vec = fake_target - missile_m1["init_pos"]
        missile_dist = np.linalg.norm(missile_vec)
        if missile_dist < epsilon:
            return 0.0
        missile_dir = missile_vec / missile_dist
        
        # 时间范围
        t_det = drop_delay + det_delay
        t_start = t_det
        t_end = t_det + smoke_param["valid_time"]
        t_list = np.arange(t_start, t_end + dt, dt)
        
        # 计算遮蔽时长
        valid_total = 0.0
        
        for t in t_list:
            # 导弹位置
            missile_pos = missile_m1["init_pos"] + missile_dir * missile_m1["speed"] * t
            
            # 烟幕位置
            sink_time = t - t_det
            smoke_center = np.array([
                det_point[0],
                det_point[1],
                det_point[2] - smoke_param["sink_speed"] * sink_time
            ])
            
            # 判定遮蔽状态
            if is_target_shielded(missile_pos, smoke_center, smoke_param["r"], target_samples):
                valid_total += dt
        
        return valid_total
    
    except Exception:
        return 0.0


# -------------------------- 6. 并行计算辅助函数 --------------------------
def evaluate_parameter_batch(param_batch):
    """并行计算参数批次"""
    best_time = 0.0
    best_params = None
    
    # 基础方向向量（全局变量替代）
    uav_to_target = fake_target - fy1_param["init_pos"]
    base_direction = uav_to_target[:2] / np.linalg.norm(uav_to_target[:2])
    
    for speed, drop_delay, det_delay, angle_offset in param_batch:
        # 计算飞行方向
        cos_offset = np.cos(angle_offset)
        sin_offset = np.sin(angle_offset)
        flight_direction = np.array([
            base_direction[0] * cos_offset - base_direction[1] * sin_offset,
            base_direction[0] * sin_offset + base_direction[1] * cos_offset
        ])
        
        shield_time = calculate_shield_time(speed, drop_delay, det_delay, flight_direction)
        
        if shield_time > best_time:
            best_time = shield_time
            best_params = {
                "speed": speed,
                "drop_delay": drop_delay,
                "det_delay": det_delay,
                "flight_direction": flight_direction,
                "angle_offset": angle_offset,
                "shield_time": shield_time
            }
    
    return best_params, best_time


def create_parameter_batches(param_combinations, num_processes):
    """将参数组合分割成批次用于并行处理"""
    batch_size = len(param_combinations) // num_processes
    batches = []
    
    for i in range(0, len(param_combinations), batch_size):
        batch = param_combinations[i:i + batch_size]
        if batch:  # 确保批次不为空
            batches.append(batch)
    
    return batches


# -------------------------- 7. 并行启发式搜索算法 --------------------------
def parallel_heuristic_search():
    """并行启发式搜索最优参数"""
    num_processes = cpu_count()
    print(f"开始并行启发式搜索...（使用 {num_processes} 个CPU核心）")
    
    best_params = None
    best_time = 0.0
    
    # 第一阶段：并行粗网格搜索
    print("第一阶段：并行粗网格搜索")
    speed_candidates = np.linspace(70, 140, 15)
    drop_delay_candidates = np.linspace(0.5, 5.0, 19)
    det_delay_candidates = np.linspace(1.0, 8.0, 29)
    angle_offsets = np.linspace(-np.pi/4, np.pi/4, 17)  # ±45度范围
    
    # 生成所有参数组合
    param_combinations = list(product(speed_candidates, drop_delay_candidates, 
                                    det_delay_candidates, angle_offsets))
    total_combinations = len(param_combinations)
    print(f"总参数组合数: {total_combinations}")
    
    # 创建参数批次
    param_batches = create_parameter_batches(param_combinations, num_processes)
    print(f"分割为 {len(param_batches)} 个批次进行并行处理")
    
    # 并行执行
    start_time = time.time()
    with Pool(processes=num_processes) as pool:
        results = pool.map(evaluate_parameter_batch, param_batches)
    stage1_time = time.time() - start_time
    
    # 收集结果
    for result_params, result_time in results:
        if result_params and result_time > best_time:
            best_time = result_time
            best_params = result_params
            print(f"新最优解：遮蔽时长 = {best_time:.4f}s")
            print(f"  速度: {best_params['speed']:.1f} m/s")
            print(f"  投放延时: {best_params['drop_delay']:.2f} s")
            print(f"  起爆延时: {best_params['det_delay']:.2f} s")
            print(f"  角度偏移: {np.degrees(best_params['angle_offset']):.1f}°")
    
    print(f"第一阶段完成，用时: {stage1_time:.2f}秒，最佳遮蔽时长: {best_time:.4f}s")
    
    if best_params is None:
        print("第一阶段未找到有效解")
        return None
    
    # 第二阶段：并行局部精细搜索
    print("\n第二阶段：并行局部精细搜索")
    
    # 在最佳参数附近进行精细搜索
    speed_range = [max(70, best_params["speed"] - 10), min(140, best_params["speed"] + 10)]
    drop_delay_range = [max(0.5, best_params["drop_delay"] - 0.5), min(5.0, best_params["drop_delay"] + 0.5)]
    det_delay_range = [max(1.0, best_params["det_delay"] - 0.5), min(8.0, best_params["det_delay"] + 0.5)]
    angle_range = [best_params["angle_offset"] - np.pi/12, best_params["angle_offset"] + np.pi/12]
    
    speed_fine = np.linspace(speed_range[0], speed_range[1], 21)
    drop_delay_fine = np.linspace(drop_delay_range[0], drop_delay_range[1], 21)
    det_delay_fine = np.linspace(det_delay_range[0], det_delay_range[1], 21)
    angle_fine = np.linspace(angle_range[0], angle_range[1], 13)
    
    # 生成精细搜索参数组合
    fine_combinations = list(product(speed_fine, drop_delay_fine, det_delay_fine, angle_fine))
    fine_batches = create_parameter_batches(fine_combinations, num_processes)
    print(f"精细搜索组合数: {len(fine_combinations)}, 分割为 {len(fine_batches)} 个批次")
    
    # 并行执行精细搜索
    start_time = time.time()
    with Pool(processes=num_processes) as pool:
        fine_results = pool.map(evaluate_parameter_batch, fine_batches)
    stage2_time = time.time() - start_time
    
    # 收集精细搜索结果
    for result_params, result_time in fine_results:
        if result_params and result_time > best_time:
            best_time = result_time
            best_params = result_params
            print(f"精细搜索新最优解：遮蔽时长 = {best_time:.4f}s")
    
    print(f"第二阶段完成，用时: {stage2_time:.2f}秒，最终最佳遮蔽时长: {best_time:.4f}s")
    print(f"总并行搜索用时: {stage1_time + stage2_time:.2f}秒")
    
    return best_params, best_time


# -------------------------- 8. 主程序 --------------------------
if __name__ == "__main__":
    print("="*80)
    print("问题1：固定参数计算")
    print("="*80)
    
    # 问题1：固定参数计算
    uav_to_fake = fake_target - fy1_param["init_pos"]
    fixed_direction = uav_to_fake[:2] / np.linalg.norm(uav_to_fake[:2])
    
    fixed_time = calculate_shield_time(120.0, 1.5, 3.6, fixed_direction)
    print(f"问题1结果：固定参数下的遮蔽时长 = {fixed_time:.4f}秒")
    
    print("\n" + "="*80)
    print("问题2：并行启发式搜索优化")
    print("="*80)
    
    # 问题2：并行启发式搜索
    start_time = time.time()
    result = parallel_heuristic_search()
    end_time = time.time()
    
    if result:
        best_params, best_time = result
        
        print("\n" + "="*80)
        print("最优解详细信息")
        print("="*80)
        print(f"最大遮蔽时长: {best_time:.4f} 秒")
        print(f"最优飞行速度: {best_params['speed']:.2f} m/s")
        print(f"最优投放延时: {best_params['drop_delay']:.3f} s")
        print(f"最优起爆延时: {best_params['det_delay']:.3f} s")
        print(f"飞行方向角度偏移: {np.degrees(best_params['angle_offset']):.2f}°")
        print(f"飞行方向向量: ({best_params['flight_direction'][0]:.6f}, {best_params['flight_direction'][1]:.6f})")
        
        # 计算关键位置
        drop_point = calc_drop_point(
            fy1_param["init_pos"], 
            best_params["speed"], 
            best_params["drop_delay"], 
            best_params["flight_direction"]
        )
        
        det_point = calc_det_point(
            drop_point, 
            best_params["speed"], 
            best_params["det_delay"], 
            g, 
            best_params["flight_direction"]
        )
        
        print(f"\n烟幕弹投放点: ({drop_point[0]:.2f}, {drop_point[1]:.2f}, {drop_point[2]:.2f})")
        print(f"烟幕弹起爆点: ({det_point[0]:.2f}, {det_point[1]:.2f}, {det_point[2]:.2f})")
        
        improvement = (best_time - fixed_time) / fixed_time * 100
        print(f"\n相比问题1的改进: +{improvement:.2f}%")
        print(f"总搜索用时: {end_time - start_time:.2f} 秒")
        
        # 性能分析
        sequential_estimate = (end_time - start_time) * cpu_count()
        speedup = sequential_estimate / (end_time - start_time)
        efficiency = speedup / cpu_count() * 100
        print(f"并行加速比: {speedup:.1f}x")
        print(f"并行效率: {efficiency:.1f}%")
        
    else:
        print("未找到有效解")