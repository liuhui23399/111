import csv
import math
import multiprocessing as mp
from itertools import product, combinations
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
import numpy as np

# Constants
G = 9.8  # gravity (m/s^2)
MISSILE_SPEED = 300.0  # m/s
CLOUD_RADIUS = 10.0  # m (effective radius)
CLOUD_DURATION = 20.0  # s (effective duration)
CLOUD_SINK = 3.0  # m/s downward
DT = 0.02  # simulation step (s)

# Search ranges
SPEED_MIN, SPEED_MAX = 70.0, 140.0
DROP_MIN, DROP_MAX = 0.0, 60.0
FUSE_MIN, FUSE_MAX = 0.0, 20.0

# Coarse grid
DRONE_SPEED_RANGE = range(70, 141, 10)  # 70–140 m/s step 10
DROP_TIME_RANGE = range(0, 61, 2)  # seconds
FUSE_RANGE = [f / 2.0 for f in range(0, 41)]  # 0–20 s in 0.5 increments
BEARING_RANGE = range(0, 360, 15)  # degrees

# Positions (x, y, z)
TARGET_FAKE = (0.0, 0.0, 0.0)
TRUE_TARGET = (0.0, 200.0, 0.0)

# 3枚导弹的初始位置
MISSILES = {
    "M1": (20000.0, 0.0, 2000.0),
    "M2": (19000.0, 600.0, 2100.0),
    "M3": (18000.0, -600.0, 1900.0),
}

# 5架无人机的位置
DRONES = {
    "FY1": (17800.0, 0.0, 1800.0),
    "FY2": (12000.0, 1400.0, 1400.0),
    "FY3": (6000.0, -3000.0, 700.0),
    "FY4": (11000.0, 2000.0, 1800.0),
    "FY5": (13000.0, -2000.0, 1300.0),
}

VERBOSE = True

def missile_pos(missile_start, t: float):
    """Position of missile at time t."""
    mx, my, mz = (TARGET_FAKE[i] - missile_start[i] for i in range(3))
    _norm = math.hypot(math.hypot(mx, my), mz)
    missile_dir = (mx / _norm, my / _norm, mz / _norm)
    
    return (
        missile_start[0] + missile_dir[0] * MISSILE_SPEED * t,
        missile_start[1] + missile_dir[1] * MISSILE_SPEED * t,
        missile_start[2] + missile_dir[2] * MISSILE_SPEED * t,
    )

def cover_time(speed: float, bearing_deg: float, t_drop: float, fuse: float, pos, missile_start):
    """Compute effective cover duration for a parameter set."""
    # Heading
    rad = math.radians(bearing_deg % 360)
    heading = (math.cos(rad), math.sin(rad), 0.0)

    # Drop point
    px_drop = pos[0] + heading[0] * speed * t_drop
    py_drop = pos[1] + heading[1] * speed * t_drop
    pz_drop = pos[2]  # constant altitude

    # Explosion point
    px_exp = px_drop + heading[0] * speed * fuse
    py_exp = py_drop + heading[1] * speed * fuse
    pz_exp = pz_drop - 0.5 * G * fuse * fuse
    if pz_exp < 0:
        return 0.0  # invalid, grenade underground

    t_explode = t_drop + fuse
    t_end = t_explode + CLOUD_DURATION

    cover = 0.0
    inside = False
    last_t = None
    t = t_explode
    while t <= t_end:
        mx_t, my_t, mz_t = missile_pos(missile_start, t)
        cz = pz_exp - CLOUD_SINK * (t - t_explode)

        # Distance from line segment Missile->Target to cloud centre
        vx, vy, vz = TRUE_TARGET[0] - mx_t, TRUE_TARGET[1] - my_t, TRUE_TARGET[2] - mz_t
        wx, wy, wz = px_exp - mx_t, py_exp - my_t, cz - mz_t
        v_len2 = vx * vx + vy * vy + vz * vz
        if v_len2 == 0:
            dist2_los = wx * wx + wy * wy + wz * wz
        else:
            proj = (wx * vx + wy * vy + wz * vz) / v_len2
            proj = max(0.0, min(1.0, proj))
            closest_x = mx_t + proj * vx
            closest_y = my_t + proj * vy
            closest_z = mz_t + proj * vz
            dx = px_exp - closest_x
            dy = py_exp - closest_y
            dz = cz - closest_z
            dist2_los = dx * dx + dy * dy + dz * dz

        if dist2_los <= CLOUD_RADIUS * CLOUD_RADIUS:
            if not inside:
                inside = True
                last_t = t
        else:
            if inside:
                cover += t - last_t
                inside = False
        t += DT
    if inside:
        cover += t_end + DT - last_t
    return min(cover, CLOUD_DURATION)

def collaborative_cover_time(drone_configs, missile_start):
    """计算多架无人机协同干扰的总覆盖时长"""
    total_cover = 0.0
    time_points = []
    
    # 收集所有干扰弹的时间段
    for drone_name, drone_pos, speed, bearing, t_drop, fuse in drone_configs:
        if t_drop < 0 or fuse < 0:
            continue
            
        rad = math.radians(bearing % 360)
        heading = (math.cos(rad), math.sin(rad), 0.0)
        
        px_drop = drone_pos[0] + heading[0] * speed * t_drop
        py_drop = drone_pos[1] + heading[1] * speed * t_drop
        pz_drop = drone_pos[2]
        
        px_exp = px_drop + heading[0] * speed * fuse
        py_exp = py_drop + heading[1] * speed * fuse
        pz_exp = pz_drop - 0.5 * G * fuse * fuse
        
        if pz_exp < 0:
            continue
            
        t_explode = t_drop + fuse
        t_end = t_explode + CLOUD_DURATION
        
        # 计算每个时间步的覆盖情况
        t = t_explode
        while t <= t_end:
            mx_t, my_t, mz_t = missile_pos(missile_start, t)
            cz = pz_exp - CLOUD_SINK * (t - t_explode)
            
            # 计算到导弹轨迹的距离
            vx, vy, vz = TRUE_TARGET[0] - mx_t, TRUE_TARGET[1] - my_t, TRUE_TARGET[2] - mz_t
            wx, wy, wz = px_exp - mx_t, py_exp - my_t, cz - mz_t
            v_len2 = vx * vx + vy * vy + vz * vz
            if v_len2 == 0:
                dist2_los = wx * wx + wy * wy + wz * wz
            else:
                proj = (wx * vx + wy * vy + wz * vz) / v_len2
                proj = max(0.0, min(1.0, proj))
                closest_x = mx_t + proj * vx
                closest_y = my_t + proj * vy
                closest_z = mz_t + proj * vz
                dx = px_exp - closest_x
                dy = py_exp - closest_y
                dz = cz - closest_z
                dist2_los = dx * dx + dy * dy + dz * dz
            
            if dist2_los <= CLOUD_RADIUS * CLOUD_RADIUS:
                time_points.append((t, 1))  # 被覆盖
            else:
                time_points.append((t, 0))  # 未被覆盖
            t += DT
    
    # 按时间排序并计算总覆盖时长
    time_points.sort(key=lambda x: x[0])
    covered = False
    start_time = None
    
    for t, is_covered in time_points:
        if is_covered and not covered:
            covered = True
            start_time = t
        elif not is_covered and covered:
            total_cover += t - start_time
            covered = False
    
    if covered and start_time is not None:
        total_cover += time_points[-1][0] - start_time
    
    return min(total_cover, CLOUD_DURATION * len(drone_configs))

def optimize_collaborative_strategy():
    """协同优化策略：为每枚导弹分配多架无人机"""
    print("开始协同优化：多架无人机协同干扰策略...")
    
    all_results = []
    
    # 为每枚导弹分配无人机组合
    missile_assignments = {
        "M1": ["FY1", "FY2"],  # 2架无人机协同干扰M1
        "M2": ["FY3", "FY4"],  # 2架无人机协同干扰M2  
        "M3": ["FY5"]          # 1架无人机干扰M3
    }
    
    for missile_id, assigned_drones in missile_assignments.items():
        print(f"\n优化导弹 {missile_id} 的协同干扰策略...")
        print(f"分配无人机: {assigned_drones}")
        
        missile_start = MISSILES[missile_id]
        
        # 为每架分配的无人机生成1-3枚干扰弹
        for drone_name in assigned_drones:
            drone_pos = DRONES[drone_name]
            print(f"  处理无人机 {drone_name}...")
            
            # 生成1-3枚干扰弹
            for projectile_num in range(1, 4):
                if projectile_num == 1:
                    # 第一枚：独立优化
                    best_cov, best_params = optimize_single_drone(drone_name, drone_pos, missile_start)
                    if best_cov > 0.1:
                        result = create_result(drone_name, projectile_num, best_params, missile_id, best_cov)
                        all_results.append(result)
                        print(f"    干扰弹 {projectile_num}: 干扰时长 {best_cov:.3f}s")
                    else:
                        print(f"    干扰弹 {projectile_num}: 无法找到有效解")
                else:
                    # 后续干扰弹：基于协同效果调整
                    if any(r["无人机编号"] == drone_name and r["烟幕干扰弹编号"] == 1 for r in all_results):
                        first_bomb = next(r for r in all_results if r["无人机编号"] == drone_name and r["烟幕干扰弹编号"] == 1)
                        
                        # 调整时间间隔
                        time_interval = 6.0 + projectile_num * 2.0  # 递增间隔
                        adjusted_t_drop = first_bomb["烟幕干扰弹投放点x坐标 (m)"] / first_bomb["无人机运动速度 (m/s)"] + time_interval
                        
                        speed = first_bomb["无人机运动速度 (m/s)"]
                        bearing = first_bomb["无人机运动方向"]
                        fuse = 4.0 + projectile_num * 0.5  # 递增引信时间
                        
                        cov = cover_time(speed, bearing, adjusted_t_drop, fuse, drone_pos, missile_start)
                        
                        if cov > 0.1:
                            result = create_result_from_params(drone_name, projectile_num, speed, bearing, 
                                                            adjusted_t_drop, fuse, missile_id, cov, drone_pos)
                            all_results.append(result)
                            print(f"    干扰弹 {projectile_num}: 干扰时长 {cov:.3f}s")
                        else:
                            print(f"    干扰弹 {projectile_num}: 无法找到有效解")
    
    return all_results

def optimize_single_drone(drone_name, drone_pos, missile_start):
    """优化单架无人机的参数"""
    best_cov, best_params = -1.0, None
    
    # 粗搜索
    for speed, bearing_deg, t_drop, fuse in product(DRONE_SPEED_RANGE, BEARING_RANGE, DROP_TIME_RANGE, FUSE_RANGE):
        cov = cover_time(speed, bearing_deg, t_drop, fuse, drone_pos, missile_start)
        if cov > best_cov:
            best_cov, best_params = cov, (speed, bearing_deg, t_drop, fuse)
        if best_cov >= CLOUD_DURATION - 1e-3:
            break
    
    # 精细优化
    if best_params and best_cov < CLOUD_DURATION - 1e-3:
        def objective(x):
            speed = min(max(x[0], SPEED_MIN), SPEED_MAX)
            bearing = x[1] % 360.0
            t_drop = min(max(x[2], DROP_MIN), DROP_MAX)
            fuse = min(max(x[3], FUSE_MIN), FUSE_MAX)
            return -cover_time(speed, bearing, t_drop, fuse, drone_pos, missile_start)
        
        try:
            res = minimize(objective, x0=list(best_params), method="Nelder-Mead", 
                          options={"maxiter": 100, "xatol": 1e-2, "fatol": 1e-3})
            if res.success:
                opt_speed = min(max(res.x[0], SPEED_MIN), SPEED_MAX)
                opt_bearing = res.x[1] % 360.0
                opt_tdrop = min(max(res.x[2], DROP_MIN), DROP_MAX)
                opt_fuse = min(max(res.x[3], FUSE_MIN), FUSE_MAX)
                best_cov = cover_time(opt_speed, opt_bearing, opt_tdrop, opt_fuse, drone_pos, missile_start)
                best_params = (opt_speed, opt_bearing, opt_tdrop, opt_fuse)
        except:
            pass
    
    return best_cov, best_params

def create_result(drone_name, projectile_num, params, missile_id, coverage):
    """创建结果记录"""
    speed, bearing, t_drop, fuse = params
    drone_pos = DRONES[drone_name]
    
    rad = math.radians(bearing % 360)
    heading = (math.cos(rad), math.sin(rad), 0.0)
    
    px_drop = drone_pos[0] + heading[0] * speed * t_drop
    py_drop = drone_pos[1] + heading[1] * speed * t_drop
    pz_drop = drone_pos[2]
    px_exp = px_drop + heading[0] * speed * fuse
    py_exp = py_drop + heading[1] * speed * fuse
    pz_exp = pz_drop - 0.5 * G * fuse * fuse
    
    return {
        "无人机编号": drone_name,
        "无人机运动方向": round(bearing % 360, 2),
        "无人机运动速度 (m/s)": round(speed, 2),
        "烟幕干扰弹编号": projectile_num,
        "烟幕干扰弹投放点x坐标 (m)": round(px_drop, 2),
        "烟幕干扰弹投放点y坐标 (m)": round(py_drop, 2),
        "烟幕干扰弹投放点z坐标 (m)": round(pz_drop, 2),
        "烟幕干扰弹起爆点x坐标 (m)": round(px_exp, 2),
        "烟幕干扰弹起爆点y坐标 (m)": round(py_exp, 2),
        "烟幕干扰弹起爆点z坐标 (m)": round(pz_exp, 2),
        "有效干扰时长 (s)": round(coverage, 3),
        "干扰的导弹编号": missile_id,
    }

def create_result_from_params(drone_name, projectile_num, speed, bearing, t_drop, fuse, missile_id, coverage, drone_pos):
    """从参数创建结果记录"""
    rad = math.radians(bearing % 360)
    heading = (math.cos(rad), math.sin(rad), 0.0)
    
    px_drop = drone_pos[0] + heading[0] * speed * t_drop
    py_drop = drone_pos[1] + heading[1] * speed * t_drop
    pz_drop = drone_pos[2]
    px_exp = px_drop + heading[0] * speed * fuse
    py_exp = py_drop + heading[1] * speed * fuse
    pz_exp = pz_drop - 0.5 * G * fuse * fuse
    
    return {
        "无人机编号": drone_name,
        "无人机运动方向": round(bearing % 360, 2),
        "无人机运动速度 (m/s)": round(speed, 2),
        "烟幕干扰弹编号": projectile_num,
        "烟幕干扰弹投放点x坐标 (m)": round(px_drop, 2),
        "烟幕干扰弹投放点y坐标 (m)": round(py_drop, 2),
        "烟幕干扰弹投放点z坐标 (m)": round(pz_drop, 2),
        "烟幕干扰弹起爆点x坐标 (m)": round(px_exp, 2),
        "烟幕干扰弹起爆点y坐标 (m)": round(py_exp, 2),
        "烟幕干扰弹起爆点z坐标 (m)": round(pz_exp, 2),
        "有效干扰时长 (s)": round(coverage, 3),
        "干扰的导弹编号": missile_id,
    }

def solve_problem5_collaborative():
    """协同优化求解第五问"""
    print("开始协同优化求解第五问：5架无人机对3枚导弹的协同干扰策略...")
    
    all_results = optimize_collaborative_strategy()
    
    # 保存结果
    if all_results:
        df = pd.DataFrame(all_results)
        
        # 重新排列列顺序
        column_order = [
            "无人机编号",
            "无人机运动方向", 
            "无人机运动速度 (m/s)",
            "烟幕干扰弹编号",
            "烟幕干扰弹投放点x坐标 (m)",
            "烟幕干扰弹投放点y坐标 (m)", 
            "烟幕干扰弹投放点z坐标 (m)",
            "烟幕干扰弹起爆点x坐标 (m)",
            "烟幕干扰弹起爆点y坐标 (m)",
            "烟幕干扰弹起爆点z坐标 (m)",
            "有效干扰时长 (s)",
            "干扰的导弹编号"
        ]
        
        df = df[column_order]
        
        # 保存到Excel
        df.to_excel("result3_collaborative.xlsx", index=False, engine='openpyxl')
        print(f"\n协同优化结果已保存到 result3_collaborative.xlsx")
        print(f"总记录数: {len(all_results)}")
        print(f"总有效干扰时长: {df['有效干扰时长 (s)'].sum():.3f} 秒")
        
        # 按导弹统计
        print("\n按导弹统计干扰效果:")
        for missile_id in MISSILES.keys():
            missile_data = df[df["干扰的导弹编号"] == missile_id]
            if not missile_data.empty:
                missile_coverage = missile_data["有效干扰时长 (s)"].sum()
                drone_count = missile_data["无人机编号"].nunique()
                bomb_count = len(missile_data)
                print(f"{missile_id}: {drone_count} 架无人机, {bomb_count} 枚干扰弹, 总干扰时长 {missile_coverage:.3f} 秒")
        
        # 按无人机统计
        print("\n按无人机统计:")
        for drone in DRONES.keys():
            drone_data = df[df["无人机编号"] == drone]
            if not drone_data.empty:
                drone_coverage = drone_data["有效干扰时长 (s)"].sum()
                print(f"{drone}: {len(drone_data)} 枚干扰弹，总干扰时长 {drone_coverage:.3f} 秒")
    else:
        print("未找到有效解")

if __name__ == "__main__":
    solve_problem5_collaborative()
