import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import copy

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
    "FY3": np.array([6000.0, -3000.0, 700.0])
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


# -------------------------- 2. 复用之前的核心函数 --------------------------
def calc_uav_direction(uav_init_pos, fake_target, angle_deg):
    """计算无人机飞行方向向量（使用绝对角度）"""
    angle_rad = np.radians(angle_deg)
    dir_vec_xy = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    return dir_vec_xy

def calc_drop_point(uav_init_pos, uav_speed, drop_delay, fake_target, angle_deg):
    """计算烟幕弹投放点"""
    dir_vec_xy = calc_uav_direction(uav_init_pos, fake_target, angle_deg)
    flight_dist = uav_speed * drop_delay
    drop_xy = uav_init_pos[:2] + dir_vec_xy * flight_dist
    drop_z = uav_init_pos[2]
    return np.array([drop_xy[0], drop_xy[1], drop_z])

def calc_det_point(drop_point, uav_speed, det_delay, g, fake_target, angle_deg, uav_init_pos):
    """计算烟幕弹起爆点"""
    dir_vec_xy = calc_uav_direction(uav_init_pos, fake_target, angle_deg)
    horizontal_dist = uav_speed * det_delay
    det_xy = drop_point[:2] + dir_vec_xy * horizontal_dist
    drop_h = 0.5 * g * det_delay ** 2
    det_z = drop_point[2] - drop_h
    return np.array([det_xy[0], det_xy[1], det_z])

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

def is_segment_intersect_sphere(M, P, C, r):
    """判定线段MP与球C(r)是否相交"""
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

def is_target_shielded_multi_smoke(missile_pos, smoke_centers, smoke_r, target_samples):
    """判定真目标是否被多个烟幕中的任一个完全遮蔽"""
    for smoke_center in smoke_centers:
        # 检查单个烟幕是否完全遮蔽目标
        shielded = True
        for p in target_samples:
            if not is_segment_intersect_sphere(missile_pos, p, smoke_center, smoke_r):
                shielded = False
                break
        if shielded:
            return True
    return False


# -------------------------- 3. 多无人机优化函数 --------------------------
def calc_total_shield_time_multi_uav(params, uav_names, target_samples):
    """
    计算多架无人机的总遮蔽时间
    params: [speed1, drop_delay1, det_delay1, angle1, speed2, drop_delay2, det_delay2, angle2, ...]
    """
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
                if is_target_shielded_multi_smoke(missile_pos, active_smoke_centers, smoke_param["r"], target_samples):
                    valid_total += dt
        
        return valid_total
        
    except Exception as e:
        print(f"计算错误: {e}")
        return 0.0


def optimize_multi_uav(uav_names, target_samples):
    """优化多架无人机参数"""
    num_uavs = len(uav_names)
    
    # 参数边界：[speed, drop_delay, det_delay, angle] * num_uavs
    bounds = []
    for _ in range(num_uavs):
        bounds.extend([
            (70, 140),    # speed
            (0.1, 5.0),   # drop_delay
            (0.1, 5.0),   # det_delay
            (0, 360)      # angle
        ])
    
    def objective(params):
        return -calc_total_shield_time_multi_uav(params, uav_names, target_samples)
    
    print(f"开始优化{len(uav_names)}架无人机参数...")
    
    result = differential_evolution(
        objective,
        bounds,
        maxiter=300,
        popsize=20,
        seed=42,
        disp=True
    )
    
    return result


# -------------------------- 4. 主程序 --------------------------
if __name__ == "__main__":
    # 生成目标采样点
    target_samples = generate_high_density_samples(real_target)
    print(f"生成真目标采样点：{len(target_samples)}个")
    
    # 选择参与任务的无人机
    uav_names = ["FY1", "FY2", "FY3"]
    
    print(f"\n开始优化{len(uav_names)}架无人机的协同干扰策略...")
    print("参与无人机：", uav_names)
    for name in uav_names:
        print(f"{name}: {uav_positions[name]}")
    
    # 执行优化
    result = optimize_multi_uav(uav_names, target_samples)
    
    if result.success:
        print(f"\n优化成功！最大遮蔽时间：{-result.fun:.4f} 秒")
        
        # 解析最优参数
        optimal_params = result.x
        results_data = []
        
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
            print(f"  投放点：{drop_point.round(4)}")
            print(f"  起爆点：{det_point.round(4)}")
            print(f"  单独贡献时间：{single_shield_time:.4f} s")
        
        # 保存到Excel文件
        df = pd.DataFrame(results_data)
        output_file = "result2.xlsx"
        df.to_excel(output_file, index=False)
        print(f"\n结果已保存到 {output_file}")
        
        print(f"\n协同总遮蔽时间：{-result.fun:.4f} 秒")
        
    else:
        print("优化失败:", result.message)