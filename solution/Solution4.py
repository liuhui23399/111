import time

import functions
from scipy.optimize import differential_evolution

uav_initial_pos_1 = functions.UAV_POSITIONS['FY1']
uav_initial_pos_2 = functions.UAV_POSITIONS['FY2']
uav_initial_pos_3 = functions.UAV_POSITIONS['FY3']
missile_initial_pos = functions.MISSILE_POSITIONS['M1']


def objective_for_optimizer(params):
    # 解包参数
    (angle_degrees_1, uav_speed_1, t_release_delay_1, t_free_fall_1,
     angle_degrees_2, uav_speed_2, t_release_delay_2, t_free_fall_2,
     angle_degrees_3, uav_speed_3, t_release_delay_3, t_free_fall_3)= params

    # 设置参数
    params_1 = [angle_degrees_1, uav_speed_1, t_release_delay_1, t_free_fall_1, 0.2]
    params_2 = [angle_degrees_2, uav_speed_2, t_release_delay_2, t_free_fall_2, 0.2]
    params_3 = [angle_degrees_3, uav_speed_3, t_release_delay_3, t_free_fall_3, 0.2]

    # 得到遮挡区间
    mask_interval_1 = functions.calculate_obscuration_time(params_1, uav_initial_pos_1, missile_initial_pos)[1]
    mask_interval_2 = functions.calculate_obscuration_time(params_2, uav_initial_pos_2, missile_initial_pos)[1]
    mask_interval_3 = functions.calculate_obscuration_time(params_3, uav_initial_pos_3, missile_initial_pos)[1]

    # 处理遮挡区间
    all_intervals = [mask_interval_1, mask_interval_2, mask_interval_3]
    merged_intervals = functions.merge_intervals(all_intervals)
    total_mask_time = sum(end - start for start, end in merged_intervals)

    return -total_mask_time

# --- 运行智能优化算法 ---
if __name__ == '__main__':
    # 定义12个决策变量的边界
    bounds = [
        (0, 360),     # 无人机1飞行角度
        (70, 140),      # 无人机1速度
        (0, 8),         # 无人机1投弹延迟
        (0, 5),         # 烟雾弹1引爆延迟
        (0, 360),         # 无人机2飞行角度
        (70, 140),      # 无人机2速度
        (0, 8),         # 无人机2投弹延迟
        (0, 5),         # 烟雾弹2引爆延迟
        (0, 360),             # 无人机3飞行角度
        (70, 140),      # 无人机3飞行速度
        (0, 8),             # 无人机3投弹延迟
        (0, 5)              # 烟雾弹3引爆延迟
    ]

    print("开始运行差分进化算法求解 问题4...")
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
    best_params = result.x
    max_duration = -result.fun
    print("\n--- Starting Stage 2: Local Refinement ---")

    best_params_stage1 = result.x
    # Shrinkage factor: New range will be 10% of the original range, centered on the best result
    shrinkage_factor = 0.1

    refined_bounds = []
    for i, (low, high) in enumerate(bounds):
        center = best_params_stage1[i]
        original_range = high - low
        margin = original_range * shrinkage_factor / 2.0

        new_low = center - margin
        new_high = center + margin

        # Ensure the new bounds do not exceed the original valid limits
        new_low_clipped = max(new_low, low)
        new_high_clipped = min(new_high, high)

        refined_bounds.append((new_low_clipped, new_high_clipped))

    print("Refined search bounds for Stage 2:")
    for i, bound in enumerate(refined_bounds):
        print(f"  Param {i + 1}: ({bound[0]:.4f}, {bound[1]:.4f})")

    stage2_start_time = time.time()
    result_stage2 = differential_evolution(
        func=objective_for_optimizer,
        bounds=refined_bounds,
        strategy='best1bin',
        maxiter=250,  # Can use fewer iterations for the smaller search space
        popsize=20,
        tol=0.001,  # Increase the tolerance for a more precise result
        mutation=(0.5, 1),
        recombination=0.7,
        disp=False,
        seed=42,
        workers=-1
    )
    stage2_end_time = time.time()

    # --- Final Results Output ---
    total_end_time = time.time()
    print(f"\nStage 2 complete in {stage2_end_time - stage2_start_time:.2f} seconds.")
    # print(f"Total optimization time: {total_end_time - total_start_time:.2f} seconds.")

    best_params = result_stage2.x
    max_duration = -result_stage2.fun

    print("\n" + "=" * 30)
    print("Final Refined Optimal Strategy (Problem 4)")
    print("=" * 30)
    print(f"Maximum Total Effective Obscuration Time: {max_duration:.4f} s")