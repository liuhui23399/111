import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

import functions

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

uav_initial_pos = functions.UAV_POSITIONS['FY1']
missile_initial_pos = functions.MISSILE_POSITIONS['M1']

# 重新包装目标函数
def objective_for_optimizer(params):
    full_params = list(params) + [0.2] #调整精度
    return functions.calculate_obscuration_time(full_params, uav_initial_pos, missile_initial_pos)[0]

# --- 运行智能优化算法 ---
if __name__ == '__main__':
    bounds = [
        (170.0, 190.0), # 角度范围
        (70.0, 140.0), # 速度范围
        (1.0, 8.0), # 投弹延迟
        (1.0, 5.0) # 起爆延迟
    ]

    print("开始运行差分进化算法进行优化...")
    start_time = time.time()

    result = differential_evolution(
        func=objective_for_optimizer,
        bounds=bounds,
        strategy='best1bin',
        maxiter=1000,
        popsize=50,
        tol=0.001,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=False,
        seed=42,
        workers=-1
    )

    end_time = time.time()
    print(f"\n优化完成，耗时: {end_time - start_time:.2f} 秒")

    # --- 输出结果 ---
    best_params = result.x
    max_duration = -result.fun

    print("\n--- 最优投放策略 ---")
    print(f"最大有效遮蔽时长: {max_duration:.4f} 秒")
    print("\n最优参数组合:")
    print(f"  - 无人机飞行方向角: {best_params[0]:.4f} 度")
    print(f"  - 无人机飞行速度:   {best_params[1]:.4f} m/s")
    print(f"  - 烟幕弹投放时间:   {best_params[2]:.4f} 秒")
    print(f"  - 烟幕弹起爆延迟:   {best_params[3]:.4f} 秒")

    flight_angle_rad = np.deg2rad(best_params[0])
    uav_speed = best_params[1]
    t_drop = best_params[2]
    t_det_delay = best_params[3]
    v_uav_best = np.array([uav_speed * np.cos(flight_angle_rad), uav_speed * np.sin(flight_angle_rad), 0])
    p_drop_best = uav_initial_pos + v_uav_best * t_drop
    p_detonate_best = p_drop_best + v_uav_best * t_det_delay + np.array([0, 0, -0.5 * functions.g * t_det_delay ** 2])

    print("\n对应的策略点位:")
    print(f"  - 烟幕弹投放点: ({p_drop_best[0]:.2f}, {p_drop_best[1]:.2f}, {p_drop_best[2]:.2f})")
    print(f"  - 烟幕弹起爆点: ({p_detonate_best[0]:.2f}, {p_detonate_best[1]:.2f}, {p_detonate_best[2]:.2f})")