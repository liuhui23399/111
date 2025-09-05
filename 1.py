import numpy as np

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
    "speed": 120.0,  # 飞行速度(m/s)
    "drop_delay": 1.5,  # 受领任务到投放的时间(s)
    "det_delay": 3.6    # 投放至起爆的时间(s)
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

dt = 0.001  # 时间步长（高精度计算）


# -------------------------- 2. 核心位置计算函数 --------------------------
def calc_drop_point(uav_init_pos, uav_speed, drop_delay, fake_target):
    """计算烟幕弹投放点（无人机等高度飞行，水平指向原点）"""
    # 水平方向向量（仅xy平面）
    uav_xy = uav_init_pos[:2]
    target_xy = fake_target[:2]
    dist_xy = np.linalg.norm(target_xy - uav_xy)
    # 保护：避免除以零（理论上不会发生）
    if dist_xy < epsilon:
        dir_vec_xy = np.array([0.0, 0.0])
    else:
        dir_vec_xy = (target_xy - uav_xy) / dist_xy
    
    # 投放点计算（z坐标保持不变）
    flight_dist = uav_speed * drop_delay
    drop_xy = uav_xy + dir_vec_xy * flight_dist
    drop_z = uav_init_pos[2]  # 等高度飞行
    
    return np.array([drop_xy[0], drop_xy[1], drop_z])


def calc_det_point(drop_point, uav_speed, det_delay, g, fake_target):
    """计算烟幕弹起爆点（投放后水平沿无人机方向，竖直自由落体）"""
    # 水平方向运动（继承无人机速度方向）
    drop_xy = drop_point[:2]
    target_xy = fake_target[:2]
    dist_xy = np.linalg.norm(target_xy - drop_xy)
    if dist_xy < epsilon:
        dir_vec_xy = np.array([0.0, 0.0])
    else:
        dir_vec_xy = (target_xy - drop_xy) / dist_xy
    
    horizontal_dist = uav_speed * det_delay
    det_xy = drop_xy + dir_vec_xy * horizontal_dist
    
    # 竖直方向自由落体
    drop_h = 0.5 * g * det_delay ** 2
    det_z = drop_point[2] - drop_h
    
    return np.array([det_xy[0], det_xy[1], det_z])


# -------------------------- 3. 高密度真目标采样点生成 --------------------------
def generate_high_density_samples(target, num_circle=60, num_height=20):
    """
    生成超高密度采样点（覆盖目标所有表面和内部关键位置）
    - 每个圆周60个点（角度间隔6°）
    - 20个高度层（垂直间隔0.5m）
    - 增加内部网格点采样
    """
    samples = []
    center = target["center"]
    r = target["r"]
    h = target["h"]
    center_xy = center[:2]
    min_z = center[2]
    max_z = center[2] + h
    
    # 1. 外表面采样（高密度）
    # 1.1 底面圆周（z=min_z）
    theta = np.linspace(0, 2*np.pi, num_circle, endpoint=False)
    for th in theta:
        x = center_xy[0] + r * np.cos(th)
        y = center_xy[1] + r * np.sin(th)
        samples.append([x, y, min_z])
    
    # 1.2 顶面圆周（z=max_z）
    for th in theta:
        x = center_xy[0] + r * np.cos(th)
        y = center_xy[1] + r * np.sin(th)
        samples.append([x, y, max_z])
    
    # 1.3 侧面采样（20个高度层）
    heights = np.linspace(min_z, max_z, num_height, endpoint=True)
    for z in heights:
        for th in theta:
            x = center_xy[0] + r * np.cos(th)
            y = center_xy[1] + r * np.sin(th)
            samples.append([x, y, z])
    
    # 2. 内部网格点采样（增加判定可靠性）
    # 2.1 半径方向：0~r（5个间隔）
    radii = np.linspace(0, r, 5, endpoint=True)
    # 2.2 高度方向：0~h（10个间隔）
    inner_heights = np.linspace(min_z, max_z, 10, endpoint=True)
    # 2.3 角度方向：12个间隔
    inner_thetas = np.linspace(0, 2*np.pi, 12, endpoint=False)
    
    for z in inner_heights:
        for rad in radii:
            for th in inner_thetas:
                x = center_xy[0] + rad * np.cos(th)
                y = center_xy[1] + rad * np.sin(th)
                samples.append([x, y, z])
    
    # 3. 轴线关键点
    samples.extend([
        [center_xy[0], center_xy[1], min_z],
        [center_xy[0], center_xy[1], min_z + h/4],
        [center_xy[0], center_xy[1], min_z + h/2],
        [center_xy[0], center_xy[1], min_z + 3*h/4],
        [center_xy[0], center_xy[1], max_z]
    ])
    
    return np.unique(np.array(samples), axis=0)  # 去重


# -------------------------- 4. 高精度几何判定函数 --------------------------
def is_segment_intersect_sphere(M, P, C, r):
    """高精度判定线段MP与球C(r)是否相交"""
    MP = P - M
    MC = C - M
    
    a = np.dot(MP, MP)
    
    # 处理零长度线段（M与P重合）
    if a < epsilon:
        return np.linalg.norm(MC) <= r + epsilon  # 允许微小误差
    
    # 修正：b = -2 * MP · MC
    b = -2 * np.dot(MP, MC)
    c = np.dot(MC, MC) - r ** 2
    
    discriminant = b ** 2 - 4 * a * c
    if discriminant < -epsilon:  # 考虑数值误差
        return False
    
    # 处理接近零的判别式（避免复数计算）
    if discriminant < 0:
        discriminant = 0
    
    sqrt_d = np.sqrt(discriminant)
    s1 = (-b - sqrt_d) / (2 * a)
    s2 = (-b + sqrt_d) / (2 * a)
    
    # 判定是否存在s∈[0,1]的解（允许微小数值误差）
    return (s1 <= 1.0 + epsilon) and (s2 >= -epsilon)


def is_target_shielded(missile_pos, smoke_center, smoke_r, target_samples):
    """判定真目标是否被完全遮蔽（所有采样点均被烟幕阻挡）"""
    for p in target_samples:
        if not is_segment_intersect_sphere(missile_pos, p, smoke_center, smoke_r):
            return False
    return True


# -------------------------- 5. 主计算流程 --------------------------
if __name__ == "__main__":
    # 步骤1：计算投放点和起爆点
    drop_point = calc_drop_point(
        uav_init_pos=fy1_param["init_pos"],
        uav_speed=fy1_param["speed"],
        drop_delay=fy1_param["drop_delay"],
        fake_target=fake_target
    )
    
    det_point = calc_det_point(
        drop_point=drop_point,
        uav_speed=fy1_param["speed"],
        det_delay=fy1_param["det_delay"],
        g=g,
        fake_target=fake_target
    )
    
    print("=== 基础位置信息 ===")
    print(f"无人机初始位置：{fy1_param['init_pos'].round(4)}")
    print(f"烟幕弹投放点：{drop_point.round(4)}")
    print(f"烟幕弹起爆点：{det_point.round(4)}")
    print(f"假目标位置：{fake_target}")

    # 步骤2：生成高密度真目标采样点
    target_samples = generate_high_density_samples(real_target)
    print(f"\n=== 采样点信息 ===")
    print(f"真目标采样点总数：{len(target_samples)}（含外表面和内部点）")

    # 步骤3：导弹飞行方向（指向假目标原点）
    missile_vec = fake_target - missile_m1["init_pos"]
    missile_dist = np.linalg.norm(missile_vec)
    if missile_dist < epsilon:
        missile_dir = np.array([0.0, 0.0, 0.0])
    else:
        missile_dir = missile_vec / missile_dist
    print(f"\n=== 导弹信息 ===")
    print(f"导弹初始位置：{missile_m1['init_pos'].round(4)}")
    print(f"导弹飞行方向向量：{missile_dir.round(6)}")

    # 步骤4：时间范围定义
    t_det = fy1_param["drop_delay"] + fy1_param["det_delay"]  # 起爆时刻
    t_start = t_det
    t_end = t_det + smoke_param["valid_time"]
    t_list = np.arange(t_start, t_end + dt, dt)
    print(f"\n=== 时间范围 ===")
    print(f"起爆时刻：{t_det:.2f}s")
    print(f"有效时间窗口：[{t_start:.2f}s, {t_end:.2f}s]，共{len(t_list)}个时间步")

    # 步骤5：高精度迭代计算
    valid_total = 0.0
    valid_log = []
    prev_valid = False
    shield_segments = []  # 记录遮蔽时间段（开始-结束）

    for t in t_list:
        # 计算导弹位置
        flight_time = t
        missile_pos = missile_m1["init_pos"] + missile_dir * missile_m1["speed"] * flight_time

        # 计算烟幕位置（起爆后xy固定，z下沉）
        sink_time = t - t_det
        smoke_center = np.array([
            det_point[0],
            det_point[1],
            det_point[2] - smoke_param["sink_speed"] * sink_time
        ])

        # 判定遮蔽状态
        current_valid = is_target_shielded(missile_pos, smoke_center, smoke_param["r"], target_samples)
        
        # 记录有效时间
        if current_valid:
            valid_total += dt
            valid_log.append({
                "t": round(t, 3),
                "missile_pos": missile_pos.round(4),
                "smoke_center": smoke_center.round(4)
            })
        
        # 记录遮蔽时间段
        if current_valid and not prev_valid:
            # 新的遮蔽段开始
            shield_segments.append({"start": t})
        elif not current_valid and prev_valid:
            # 遮蔽段结束
            if shield_segments:
                shield_segments[-1]["end"] = t - dt  # 上一时刻为结束点
        
        prev_valid = current_valid

    # 处理最后一个未结束的遮蔽段
    if shield_segments and "end" not in shield_segments[-1]:
        shield_segments[-1]["end"] = t_end

    # 步骤6：输出高精度结果
    print("\n" + "="*80)
    print(f"【最终结果】真目标被有效遮蔽的总时长：{valid_total:.4f} 秒")
    print("="*80)

    # 输出遮蔽时间段分析
    print("\n=== 遮蔽时间段详情 ===")
    if not shield_segments:
        print("无有效遮蔽时间段")
    else:
        for i, seg in enumerate(shield_segments, 1):
            duration = seg["end"] - seg["start"]
            print(f"第{i}段：{seg['start']:.4f}s ~ {seg['end']:.4f}s，时长：{duration:.4f}s")

    # 输出采样时刻的状态（首尾各3个）
    print("\n=== 采样时刻状态示例 ===")
    if valid_log:
        print("前3个有效时刻：")
        for log in valid_log[:3]:
            print(f"t={log['t']}s | 导弹位置：{log['missile_pos']} | 烟幕中心：{log['smoke_center']}")
        
        print("\n最后3个有效时刻：")
        for log in valid_log[-3:]:
            print(f"t={log['t']}s | 导弹位置：{log['missile_pos']} | 烟幕中心：{log['smoke_center']}")
    else:
        print("无有效遮蔽时刻")