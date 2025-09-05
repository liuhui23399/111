import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

# 复用1.py中的常量定义
g = 9.80665  # 重力加速度 (m/s²)
epsilon = 1e-12  # 数值计算保护阈值

# 目标定义
fake_target = np.array([0.0, 0.0, 0.0])
real_target = {
    "center": np.array([0.0, 200.0, 0.0]),
    "r": 7.0,
    "h": 10.0
}

# 烟幕参数
smoke_param = {
    "r": 10.0,
    "sink_speed": 3.0,
    "valid_time": 20.0
}

# 导弹M1参数
missile_m1 = {
    "init_pos": np.array([20000.0, 0.0, 2000.0]),
    "speed": 300.0
}

# 无人机FY1初始参数
fy1_param = {
    "init_pos": np.array([17800.0, 0.0, 1800.0]),
}

dt = 0.01  # 时间步长

# 核心函数
def calc_drop_point(uav_init_pos, uav_dir, uav_speed, drop_delay):
    """计算烟幕弹投放点（基于无人机飞行方向）"""
    flight_dist = uav_speed * drop_delay
    drop_point = uav_init_pos + uav_dir * flight_dist
    return drop_point

def calc_det_point(drop_point, uav_dir, uav_speed, det_delay, g):
    """计算烟幕弹起爆点"""
    # 水平方向运动（继承无人机方向）
    horizontal_dist = uav_speed * det_delay
    det_xy = drop_point[:2] + uav_dir[:2] * horizontal_dist
    
    # 竖直方向自由落体
    drop_h = 0.5 * g * det_delay ** 2
    det_z = drop_point[2] - drop_h
    
    return np.array([det_xy[0], det_xy[1], det_z])

def generate_high_density_samples(target, num_circle=20, num_height=8):
    """生成目标采样点（简化版本以加快优化速度）"""
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
    """线段-球相交判定"""
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
    """判定目标是否被遮蔽"""
    for p in target_samples:
        if not is_segment_intersect_sphere(missile_pos, p, smoke_center, smoke_r):
            return False
    return True

def evaluate_shielding(params):
    """评估遮蔽时间（添加调试信息）"""
    dir_angle, speed, drop_delay, det_delay = params
    
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
    t_list = np.arange(t_start, t_end + dt, dt)
    
    # 生成目标采样点
    target_samples = generate_high_density_samples(real_target)
    
    # 计算遮蔽时间
    valid_total = 0.0
    
    # 简化计算：检查关键时刻点
    check_points = np.linspace(t_start, t_end, 20)
    
    for t in check_points:
        missile_pos = missile_m1["init_pos"] + missile_dir * missile_m1["speed"] * t
        sink_time = t - t_det
        smoke_center = np.array([
            det_point[0],
            det_point[1],
            det_point[2] - smoke_param["sink_speed"] * sink_time
        ])
        
        if is_target_shielded(missile_pos, smoke_center, smoke_param["r"], target_samples):
            valid_total += 1.0  # 每个关键点贡献1分
    
    return -valid_total  # 负值用于最小化问题

def evaluate_shielding_precise(params):
    """更精确的遮蔽时间评估函数"""
    dir_angle, speed, drop_delay, det_delay = params
    
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
    
    # 使用更细密的时间步长进行评估
    fine_dt = 0.001
    t_list = np.arange(t_start, t_end + fine_dt, fine_dt)
    
    # 生成高密度目标采样点
    target_samples = generate_high_density_samples(real_target, num_circle=40, num_height=15)
    
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
            valid_total += fine_dt
    
    return -valid_total  # 负值用于最小化问题

# 验证问题1的已知解是否有效
def verify_problem1_solution():
    """验证问题1的解是否能在当前代码中产生非零遮蔽时间"""
    # 问题1的参数
    p1_dir = np.array([-1.0, 0.0, 0.0])  # 指向原点的方向
    p1_dir_angle = np.arctan2(0.0, -1.0)
    p1_speed = 120.0
    p1_drop_delay = 1.5
    p1_det_delay = 3.6
    
    p1_params = [p1_dir_angle, p1_speed, p1_drop_delay, p1_det_delay]
    p1_score = -evaluate_shielding(p1_params)
    
    print("\n=== 问题1解决方案验证 ===")
    print(f"方向角: {p1_dir_angle:.4f} rad ({np.degrees(p1_dir_angle):.2f}°)")
    print(f"速度: {p1_speed:.2f} m/s")
    print(f"投放延迟: {p1_drop_delay:.2f} s")
    print(f"起爆延迟: {p1_det_delay:.2f} s")
    print(f"遮蔽得分: {p1_score:.2f}")
    
    return p1_score > 0

# 网格搜索找到好的初始解
def grid_search():
    """执行粗粒度网格搜索找到有希望的参数区域"""
    best_score = -float('inf')
    best_params = None
    
    # 定义搜索网格
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    speeds = [70, 90, 110, 130, 140]
    drop_delays = [0.5, 1.5, 3.0, 5.0, 7.0, 10.0]
    det_delays = [1.0, 2.0, 3.6, 5.0, 7.5, 10.0]
    
    print(f"开始网格搜索，共{len(angles)*len(speeds)*len(drop_delays)*len(det_delays)}个参数组合...")
    
    for angle in angles:
        for speed in speeds:
            for drop_delay in drop_delays:
                for det_delay in det_delays:
                    params = [angle, speed, drop_delay, det_delay]
                    score = -evaluate_shielding(params)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        print(f"找到更好的参数: 方向={np.degrees(angle):.1f}°, 速度={speed:.1f}, "
                              f"投放={drop_delay:.1f}s, 起爆={det_delay:.1f}s, 得分={score:.2f}")
    
    return best_params, best_score

# 多阶段优化策略
def multi_stage_optimization():
    """多阶段优化策略"""
    print("=== 多阶段优化开始 ===")
    
    # 第一阶段：粗网格搜索
    print("第一阶段：扩展网格搜索...")
    best_score = -float('inf')
    best_params = None
    
    # 更精细的搜索网格
    angles = np.linspace(0, 2*np.pi, 24, endpoint=False)  # 增加角度精度
    speeds = np.linspace(70, 140, 15)  # 更细密的速度采样
    drop_delays = np.linspace(0.1, 8.0, 20)  # 更多投放延迟选项
    det_delays = np.linspace(0.5, 8.0, 20)   # 更多起爆延迟选项
    
    total_combinations = len(angles) * len(speeds) * len(drop_delays) * len(det_delays)
    print(f"搜索空间: {total_combinations} 个参数组合")
    
    count = 0
    for angle in angles:
        for speed in speeds:
            for drop_delay in drop_delays:
                for det_delay in det_delays:
                    count += 1
                    if count % 10000 == 0:
                        print(f"进度: {count}/{total_combinations} ({100*count/total_combinations:.1f}%)")
                    
                    params = [angle, speed, drop_delay, det_delay]
                    score = -evaluate_shielding(params)  # 使用快速评估
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        print(f"新最优: 角度={np.degrees(angle):.1f}°, 速度={speed:.1f}, "
                              f"投放={drop_delay:.2f}s, 起爆={det_delay:.2f}s, 得分={score:.2f}")
    
    print(f"第一阶段完成，最优得分: {best_score:.2f}")
    
    # 第二阶段：差分进化优化
    print("\n第二阶段：差分进化精细优化...")
    bounds = [
        (0, 2*np.pi),
        (70.0, 140.0),
        (0.1, 10.0),
        (0.1, 10.0)
    ]
    
    result = differential_evolution(
        evaluate_shielding_precise,  # 使用精确评估函数
        bounds,
        popsize=30,
        maxiter=100,
        disp=True,
        tol=0.0001,
        updating='deferred',
        workers=-1,
        x0=best_params if best_params else None,
        seed=42  # 固定随机种子以获得可重现结果
    )
    
    stage2_params = result.x
    stage2_score = -result.fun
    print(f"第二阶段完成，优化得分: {stage2_score:.4f}")
    
    # 第三阶段：局部精细调优
    print("\n第三阶段：局部精细调优...")
    
    # 在最优解周围进行局部搜索
    def local_objective(x):
        return evaluate_shielding_precise(x)
    
    # 设置更紧的边界（围绕当前最优解）
    angle_range = 0.2  # ±0.2弧度
    speed_range = 10.0  # ±10 m/s
    delay_range = 1.0   # ±1秒
    
    local_bounds = [
        (max(0, stage2_params[0] - angle_range), min(2*np.pi, stage2_params[0] + angle_range)),
        (max(70, stage2_params[1] - speed_range), min(140, stage2_params[1] + speed_range)),
        (max(0.1, stage2_params[2] - delay_range), min(10, stage2_params[2] + delay_range)),
        (max(0.1, stage2_params[3] - delay_range), min(10, stage2_params[3] + delay_range))
    ]
    
    local_result = minimize(
        local_objective,
        stage2_params,
        method='L-BFGS-B',
        bounds=local_bounds,
        options={'ftol': 1e-9, 'gtol': 1e-9, 'maxiter': 1000}
    )
    
    final_params = local_result.x
    final_score = -local_result.fun
    print(f"第三阶段完成，最终得分: {final_score:.4f}")
    
    return final_params, final_score

# 主函数
if __name__ == "__main__":
    print("开始优化无人机FY1的烟幕投放策略...\n")
    
    # 首先验证问题1的解是否有效
    problem1_works = verify_problem1_solution()
    if not problem1_works:
        print("警告：问题1的解在当前代码中不产生有效遮蔽，请检查计算函数！")
    
    # 执行多阶段优化
    start_time = time.time()
    final_params, final_score = multi_stage_optimization()
    
    # 解析最优参数
    opt_angle, opt_speed, opt_drop_delay, opt_det_delay = final_params
    opt_dir = np.array([np.cos(opt_angle), np.sin(opt_angle), 0.0])
    
    # 计算最优解下的投放点和起爆点
    opt_drop_point = calc_drop_point(
        fy1_param["init_pos"], opt_dir, opt_speed, opt_drop_delay
    )
    
    opt_det_point = calc_det_point(
        opt_drop_point, opt_dir, opt_speed, opt_det_delay, g
    )
    
    print("\n" + "="*80)
    print(f"【多阶段优化完成】总耗时: {time.time() - start_time:.2f}秒")
    print("="*80)
    print(f"最优参数:")
    print(f"飞行方向: ({opt_dir[0]:.6f}, {opt_dir[1]:.6f}, 0.0000)")
    print(f"飞行角度: {opt_angle:.6f} rad = {np.degrees(opt_angle):.3f}°")
    print(f"飞行速度: {opt_speed:.6f} m/s")
    print(f"投放延迟: {opt_drop_delay:.6f} s")
    print(f"起爆延迟: {opt_det_delay:.6f} s")
    print(f"烟幕投放点: [{opt_drop_point[0]:.6f}, {opt_drop_point[1]:.6f}, {opt_drop_point[2]:.6f}]")
    print(f"烟幕起爆点: [{opt_det_point[0]:.6f}, {opt_det_point[1]:.6f}, {opt_det_point[2]:.6f}]")
    print(f"优化得分: {final_score:.6f} 秒")
    print("="*80)
    
    # 进行超高精度最终验证
    print("\n开始进行超高精度最终验证...")
    ultra_fine_dt = 0.0001  # 更小的时间步长
    t_det = opt_drop_delay + opt_det_delay
    t_start = t_det
    t_end = t_det + smoke_param["valid_time"]
    t_list = np.arange(t_start, t_end + ultra_fine_dt, ultra_fine_dt)
    
    # 使用超高密度采样点进行最终验证
    target_samples = generate_high_density_samples(real_target, num_circle=80, num_height=25)
    valid_total = 0.0
    prev_valid = False
    shield_segments = []
    
    missile_vec = fake_target - missile_m1["init_pos"]
    missile_dir = missile_vec / np.linalg.norm(missile_vec)
    
    print(f"验证时间范围: {t_start:.4f}s ~ {t_end:.4f}s")
    print(f"时间步长: {ultra_fine_dt:.6f}s")
    print(f"目标采样点数: {len(target_samples)}")
    
    for i, t in enumerate(t_list):
        if i % 50000 == 0:  # 每5万次循环显示一次进度
            progress = 100 * i / len(t_list)
            print(f"验证进度: {progress:.1f}%")
        
        missile_pos = missile_m1["init_pos"] + missile_dir * missile_m1["speed"] * t
        
        sink_time = t - t_det
        smoke_center = np.array([
            opt_det_point[0],
            opt_det_point[1],
            opt_det_point[2] - smoke_param["sink_speed"] * sink_time
        ])
        
        current_valid = is_target_shielded(missile_pos, smoke_center, smoke_param["r"], target_samples)
        
        if current_valid:
            valid_total += ultra_fine_dt
        
        # 记录遮蔽时间段
        if current_valid and not prev_valid:
            shield_segments.append({"start": t})
        elif not current_valid and prev_valid:
            if shield_segments:
                shield_segments[-1]["end"] = t - ultra_fine_dt
        
        prev_valid = current_valid
    
    # 处理最后一个未结束的遮蔽段
    if shield_segments and "end" not in shield_segments[-1]:
        shield_segments[-1]["end"] = t_end
    
    print("\n最终超高精度验证结果:")
    print(f"最终精确计算的有效遮蔽总时长: {valid_total:.6f} 秒")
    
    # 输出遮蔽时间段详情
    print("\n=== 遮蔽时间段详情 ===")
    if not shield_segments:
        print("无有效遮蔽时间段")
    else:
        total_duration = 0
        for i, seg in enumerate(shield_segments, 1):
            duration = seg["end"] - seg["start"]
            total_duration += duration
            print(f"第{i}段：{seg['start']:.6f}s ~ {seg['end']:.6f}s，时长：{duration:.6f}s")
        print(f"总遮蔽时长: {total_duration:.6f}s")
    
    print(f"\n最终结论: 通过多阶段优化，获得的最大有效遮蔽时间为 {valid_total:.6f} 秒")