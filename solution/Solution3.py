import time
import numpy as np
from scipy.optimize import differential_evolution
from openpyxl import load_workbook, Workbook
import functions

uav_initial_pos = functions.UAV_POSITIONS['FY1']
missile_initial_pos = functions.MISSILE_POSITIONS['M1']

def objective_for_optimizer(params):
    # 解包决策变量
    angle_degrees, uav_speed, t_release_delay_1, t_release_delay_2, t_release_delay_3, t_free_fallree_fall_1, t_free_fallree_fall_2, t_free_fallree_fall_3 = params

    # 时间处理：依次累加得到绝对投放时刻
    t_release_1 = t_release_delay_1
    t_release_2 = t_release_1 + t_release_delay_2
    t_release_3 = t_release_2 + t_release_delay_3

    # 设置参数
    params_1 = [angle_degrees, uav_speed, t_release_1, t_free_fallree_fall_1, 0.2]
    params_2 = [angle_degrees, uav_speed, t_release_2, t_free_fallree_fall_2, 0.2]
    params_3 = [angle_degrees, uav_speed, t_release_3, t_free_fallree_fall_3, 0.2]

    # 得到遮蔽区间
    res_1 = functions.calculate_obscuration_time(params_1, uav_initial_pos, missile_initial_pos)
    res_2 = functions.calculate_obscuration_time(params_2, uav_initial_pos, missile_initial_pos)
    res_3 = functions.calculate_obscuration_time(params_3, uav_initial_pos, missile_initial_pos)

    mask_interval_1 = res_1[1]
    mask_interval_2 = res_2[1]
    mask_interval_3 = res_3[1]

    # 处理遮蔽区间
    all_intervals = [mask_interval_1, mask_interval_2, mask_interval_3]
    merged_intervals = functions.merge_intervals(all_intervals)
    total_mask_time = sum(end - start for start, end in merged_intervals)

    # 返回目标函数值
    return -total_mask_time

# --- 运行智能优化算法 ---
if __name__ == '__main__':
    # 定义8个决策变量的边界
    bounds = [
        (170, 190),  # 飞行角度
        (70, 140),  # 无人机速度
        (1, 8),  # 第1枚弹投放时间
        (1, 5),  # 第2枚弹投放间隔 (最小为1s)
        (1, 5),  # 第3枚弹投放间隔 (最小为1s)
        (1, 8),  # 引信时间1
        (1, 8),  # 引信时间2
        (1, 8)  # 引信时间3
    ]

    print("开始运行差分进化算法求解 问题3...")
    start_time = time.time()

    result = differential_evolution(
        func=objective_for_optimizer,
        bounds=bounds,
        strategy='best1bin',
        maxiter=500,
        popsize=20,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=False,
        seed=42,
        workers=-1  # 使用所有CPU核心并行计算
    )

    end_time = time.time()
    print(f"\n优化完成，耗时: {end_time - start_time:.2f} 秒")

    # --- 输出结果 ---
    best_params = result.x
    max_duration = -result.fun

    print("\n" + "=" * 25)
    print("最优投放策略 (问题3)")
    print("=" * 25)
    print(f"最大总有效遮蔽时长: {max_duration.item():.4f} 秒")

    # 解析最优参数
    angle_degrees, uav_speed, t_release_delay_1, t_release_delay_2, t_release_delay_3, t_free_fall_1, t_free_fall_2, t_free_fall_3 = best_params
    t_release_1 = t_release_delay_1
    t_release_2 = t_release_1 + t_release_delay_2
    t_release_3 = t_release_2 + t_release_delay_3

    print("\n--- 无人机飞行策略 ---")
    print(f"  - 飞行方向角: {angle_degrees:.4f} 度")
    print(f"  - 飞行速度:   {uav_speed:.4f} m/s")

    # 计算分别的遮蔽时间
    _, interval_1, _, _ = functions.calculate_obscuration_time(
        [angle_degrees, uav_speed, t_release_1, t_free_fall_1, 0.2], uav_initial_pos, missile_initial_pos
    )
    _, interval_2, _, _ = functions.calculate_obscuration_time(
        [angle_degrees, uav_speed, t_release_2, t_free_fall_2, 0.2], uav_initial_pos, missile_initial_pos
    )
    _, interval_3, _, _ = functions.calculate_obscuration_time(
        [angle_degrees, uav_speed, t_release_3, t_free_fall_3, 0.2], uav_initial_pos, missile_initial_pos
    )

    print(f"  - 烟雾弹1有效时间: {interval_1[1] - interval_1[0]:.4f} s")
    print(f"  - 烟雾弹2有效时间: {interval_2[1] - interval_2[0]:.4f} s")
    print(f"  - 烟雾弹3有效时间: {interval_3[1] - interval_3[0]:.4f} s")

    # 计算最终策略点位
    flight_angle_rad = np.deg2rad(angle_degrees)
    v_uav_best = np.array([uav_speed * np.cos(flight_angle_rad), uav_speed * np.sin(flight_angle_rad), 0])

    p_d1 = uav_initial_pos + v_uav_best * t_release_1
    p_d2 = uav_initial_pos + v_uav_best * t_release_2
    p_d3 = uav_initial_pos + v_uav_best * t_release_3

    p_det1 = p_d1 + v_uav_best * t_free_fall_1 + np.array([0, 0, -0.5 * functions.g * t_free_fall_1 ** 2])
    p_det2 = p_d2 + v_uav_best * t_free_fall_2 + np.array([0, 0, -0.5 * functions.g * t_free_fall_2 ** 2])
    p_det3 = p_d3 + v_uav_best * t_free_fall_3 + np.array([0, 0, -0.5 * functions.g * t_free_fall_3 ** 2])

    bombs_info = [
        {'id': 1, 't_release': t_release_1, 't_free_fall': t_free_fall_1, 'p_drop': p_d1, 'p_det': p_det1},
        {'id': 2, 't_release': t_release_2, 't_free_fall': t_free_fall_2, 'p_drop': p_d2, 'p_det': p_det2},
        {'id': 3, 't_release': t_release_3, 't_free_fall': t_free_fall_3, 'p_drop': p_d3, 'p_det': p_det3},
    ]

    print("\n--- 烟幕弹投放策略详情 ---")
    for info in bombs_info:
        print(f"\n  [烟幕弹 {info['id']}]")
        print(f"    投放时间: {info['t_release']:.4f} s")
        print(f"    引信时间: {info['t_free_fall']:.4f} s")
        print(f"    绝对起爆时间: {info['t_release'] + info['t_free_fall']:.4f} s")
        print(f"    投放点: ({info['p_drop'][0]:.4f}, {info['p_drop'][1]:.4f}, {info['p_drop'][2]:.4f})")
        print(f"    起爆点: ({info['p_det'][0]:.4f}, {info['p_det'][1]:.4f}, {info['p_det'][2]:.4f})")





