from math import cos, sin
import numpy as np

# 参数
g = 9.8  # m/s^2
missile_speed = 300.0  # m/s
cloud_downspeed = 3.0  # m/s 云团中心下沉速度
cloud_effective_radius = 10.0  # m 有效遮蔽半径
cloud_effective_duration = 20.0  # s 有效遮蔽时间长度
cylinder_radius = 7.0   # m 圆柱体半径
cylinder_height = 10.0  # m 圆柱体高

# 初始位置
MISSILE_POSITIONS = {
    'M1': np.array([20000.0, 0.0, 2000.0]),
    'M2': np.array([19000.0, 600.0, 2100.0]),
    'M3': np.array([18000.0, -600.0, 1900.0])
}

UAV_POSITIONS = {
    'FY1': np.array([17800.0, 0.0, 1800.0]),
    'FY2': np.array([12000.0, 1400.0, 1400.0]),
    'FY3': np.array([6000.0, -3000.0, 700.0]),
    'FY4': np.array([11000.0, 2000.0, 1800.0]),
    'FY5': np.array([13000.0, -2000.0, 1300.0])
}
true_target = np.array([0.0, 200.0, 0.0])  # 真目标中心
fake_target = np.array([0.0, 0.0, 0.0])  # 假目标

# 计算导弹坐标
def calculate_missile_positon(missile_initial_pos, t):
    u_m = (fake_target - missile_initial_pos) / np.linalg.norm(fake_target - missile_initial_pos)
    return missile_initial_pos + missile_speed * u_m * t

# 计算云团中心坐标
def calculate_cloud_center(uav_initial_pos, angle_degrees, uav_speed, t_release_delay, t_free_fall, t):
    angle_radians = np.radians(angle_degrees)
    uav_velocity = np.array([cos(angle_radians) * uav_speed, sin(angle_radians) * uav_speed, 0.0])
    release_point = uav_initial_pos + uav_velocity * t_release_delay
    explode_point = release_point + uav_velocity * t_free_fall + [0.0, 0.0, -0.5 * g * t_free_fall**2]
    dt = t - t_release_delay - t_free_fall
    return explode_point + np.array([0.0, 0.0, -cloud_downspeed * dt])

# 计算点到线段距离
def point_to_segment_distance(P, A, B):
    AP = P - A
    AB = B - A
    ab2 = np.dot(AB, AB)
    if ab2 == 0.0:
        return np.linalg.norm(AP), 0.0
    t = np.dot(AP, AB) / ab2
    t_clamped = np.clip(t, 0.0, 1.0)
    closest = A + t_clamped * AB
    return np.linalg.norm(P - closest), t_clamped

# 计算特征点的坐标
def calculate_projection_key_points(missile_initial_pos, t):
    missile_positon = calculate_missile_positon(t, missile_initial_pos)
    p_bottom_center = np.array([0, 200, 0.0])
    p_top_center = np.array([0, 200, cylinder_height])
    vector_axis_to_missile_xy = missile_positon[:2] - p_bottom_center[:2]
    if np.linalg.norm(vector_axis_to_missile_xy) == 0:
        front_dir_xy = np.array([1.0, 0.0])
    else:
        front_dir_xy = vector_axis_to_missile_xy / np.linalg.norm(vector_axis_to_missile_xy)
    side_dir_xy = np.array([-front_dir_xy[1], front_dir_xy[0]])
    front_dir_3d = np.append(front_dir_xy, 0)
    side_dir_3d = np.append(side_dir_xy, 0)
    key_points = np.array([
        p_bottom_center - cylinder_radius * front_dir_3d,
        p_bottom_center - cylinder_radius * side_dir_3d,
        p_bottom_center + cylinder_radius * side_dir_3d,
        p_top_center - cylinder_radius * front_dir_3d,
        p_top_center - cylinder_radius * side_dir_3d,
        p_top_center + cylinder_radius * side_dir_3d
    ])
    return key_points

# 计算遮蔽时长
def calculate_obscuration_time(params, uav_initial_pos, missile_initial_pos):
    # 1. 参数列表
    angle_degrees, uav_speed, t_release_delay, t_free_fall, precise = params
    t_detonate_abs = t_release_delay + t_free_fall

    # 2. 模拟和计算遮蔽时长
    time_step = precise  # 使用可变的的步长以平衡速度和精度

    t_start = t_detonate_abs
    t_end = t_detonate_abs + cloud_effective_duration

    times = np.arange(t_start, t_end, time_step)
    if len(times) == 0:
        return 0, []

    mask = np.zeros_like(times, dtype=bool)
    # 用于绘图，记录每个时间点上，云团到6条视线中最远的那条的距离
    max_distances_to_los = np.zeros_like(times)

    for i, t in enumerate(times):
        cloud_center = calculate_cloud_center(uav_initial_pos, angle_degrees, uav_speed, t_release_delay, t_free_fall, t)
        missile_positon = calculate_missile_positon(missile_initial_pos, t)
        projection_key_points = calculate_projection_key_points(missile_positon, t)

        is_all_points_obscured = True  # 先假设所有点都被遮挡
        distances_this_step = []

        for point in projection_key_points:
            distance, projection_t = point_to_segment_distance(cloud_center, missile_positon, point)
            distances_this_step.append(distance)

            # 只要有一条视线不满足遮蔽条件，就判定为无效
            if not (distance <= cloud_effective_radius):
                is_all_points_obscured = False

        if is_all_points_obscured:
            mask[i] = True
        # 计算云团到6条视线中的最远距离
        if distances_this_step:
            max_distances_to_los[i] = np.max(distances_this_step)
        else:
            max_distances_to_los[i] = np.inf

    # 计算遮挡总时间
    total_mask_time = np.sum(mask) * time_step

    # 计算遮挡区间
    interval = []
    if total_mask_time > 0:
        # 通过填充和差分找到区间的开始和结束索引
        padded_mask = np.concatenate(([False], mask, [False]))
        diffs = np.diff(padded_mask.astype(int))

        start_indices = np.where(diffs == 1)[0]
        end_indices = np.where(diffs == -1)[0]

        for start_idx, end_idx in zip(start_indices, end_indices):
            start_time = times[start_idx]
            # 结束时间点应为区间最后一个有效步的结束时刻
            end_time = times[end_idx - 1] + time_step
            interval = (start_time, end_time)

    #返回结果
    return -total_mask_time, interval, max_distances_to_los, times

# 辅助函数：用于合并一系列可能重叠的时间区间。
def merge_intervals(intervals):
    valid_intervals = []
    for interval in intervals:
        if interval:
            valid_intervals.append(interval)

    # 如果过滤后列表为空，直接返回空列表
    if not valid_intervals:
        return []
    if len(valid_intervals) == 1:
        return [valid_intervals[0]]
    merged = [valid_intervals[0]]
    for current_start, current_end in valid_intervals[1:]:
        last_start, last_end = merged[-1]

        # 如果当前区间与前一个合并后的区间有重叠，则合并它们
        if current_start < last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))

    return merged

