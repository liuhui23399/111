import numpy as np
from scipy.optimize import differential_evolution
import time
import pandas as pd

# 复用2.py中的所有常量和基础函数
# ... (保持2.py中的所有常量定义和基础函数不变)

def calc_multiple_missiles_shielding(params):
    """
    计算3枚烟幕弹的总遮蔽时间
    params: [dir_angle, speed, drop_delay1, det_delay1, drop_delay2, det_delay2, drop_delay3, det_delay3]
    """
    dir_angle, speed = params[:2]
    drop_delays = params[2:5]  # 3个投放延迟
    det_delays = params[5:8]   # 3个起爆延迟
    
    # 确保投放间隔至少1秒
    sorted_drops = sorted(drop_delays)
    if (sorted_drops[1] - sorted_drops[0] < 1.0 or 
        sorted_drops[2] - sorted_drops[1] < 1.0):
        return 0  # 不满足间隔约束
    
    # 方向向量
    dir_xy = np.array([np.cos(dir_angle), np.sin(dir_angle)])
    uav_dir = np.array([dir_xy[0], dir_xy[1], 0.0])
    
    # 导弹轨迹
    missile_vec = fake_target - missile_m1["init_pos"]
    missile_dir = missile_vec / np.linalg.norm(missile_vec)
    
    # 计算每枚弹的起爆点和有效时间窗口
    smoke_clouds = []
    for i in range(3):
        drop_point = calc_drop_point(
            fy1_param["init_pos"], uav_dir, speed, drop_delays[i]
        )
        det_point = calc_det_point(
            drop_point, uav_dir, speed, det_delays[i], g
        )
        
        t_det = drop_delays[i] + det_delays[i]
        t_start = t_det
        t_end = t_det + smoke_param["valid_time"]
        
        smoke_clouds.append({
            'det_point': det_point,
            't_start': t_start,
            't_end': t_end,
            't_det': t_det
        })
    
    # 计算总的时间范围
    global_t_start = min(cloud['t_start'] for cloud in smoke_clouds)
    global_t_end = max(cloud['t_end'] for cloud in smoke_clouds)
    
    # 高精度时间采样
    fine_dt = 0.001
    t_list = np.arange(global_t_start, global_t_end + fine_dt, fine_dt)
    target_samples = generate_high_density_samples(real_target, num_circle=30, num_height=12)
    
    total_shielded_time = 0.0
    
    for t in t_list:
        missile_pos = missile_m1["init_pos"] + missile_dir * missile_m1["speed"] * t
        
        # 检查是否被任意一个烟幕云遮蔽
        is_shielded_by_any = False
        
        for cloud in smoke_clouds:
            if cloud['t_start'] <= t <= cloud['t_end']:
                sink_time = t - cloud['t_det']
                smoke_center = np.array([
                    cloud['det_point'][0],
                    cloud['det_point'][1],
                    cloud['det_point'][2] - smoke_param["sink_speed"] * sink_time
                ])
                
                if is_target_shielded(missile_pos, smoke_center, smoke_param["r"], target_samples):
                    is_shielded_by_any = True
                    break
        
        if is_shielded_by_any:
            total_shielded_time += fine_dt
    
    return total_shielded_time

def optimize_three_missiles():
    """优化3枚烟幕弹的投放策略"""
    print("=== 开始优化3枚烟幕弹投放策略 ===")
    
    # 参数边界：[角度, 速度, 投放延迟1, 起爆延迟1, 投放延迟2, 起爆延迟2, 投放延迟3, 起爆延迟3]
    bounds = [
        (0, 2*np.pi),      # 飞行角度
        (70.0, 140.0),     # 飞行速度
        (0.1, 8.0),        # 投放延迟1
        (0.1, 8.0),        # 起爆延迟1
        (0.1, 8.0),        # 投放延迟2
        (0.1, 8.0),        # 起爆延迟2
        (0.1, 8.0),        # 投放延迟3
        (0.1, 8.0),        # 起爆延迟3
    ]
    
    def objective(params):
        return -calc_multiple_missiles_shielding(params)
    
    # 第一阶段：粗搜索
    print("第一阶段：粗网格搜索...")
    best_score = 0
    best_params = None
    
    # 基于问题2的结果作为起始点的候选
    candidate_angles = [np.pi, np.radians(178.5), np.radians(180), np.radians(175)]
    candidate_speeds = [80, 90, 100, 110, 120]
    
    for angle in candidate_angles:
        for speed in candidate_speeds:
            # 尝试不同的投放时序组合
            drop_sequences = [
                [0.5, 2.0, 4.0],   # 早期密集投放
                [1.0, 3.0, 6.0],   # 均匀分布
                [2.0, 4.0, 7.0],   # 中后期投放
                [0.5, 3.5, 7.5],   # 前后分布
            ]
            
            for drops in drop_sequences:
                for det1 in [1.0, 2.0, 3.0]:
                    for det2 in [1.5, 2.5, 3.5]:
                        for det3 in [2.0, 3.0, 4.0]:
                            params = [angle, speed] + drops + [det1, det2, det3]
                            score = calc_multiple_missiles_shielding(params)
                            
                            if score > best_score:
                                best_score = score
                                best_params = params
                                print(f"新最优: 得分={score:.4f}s, 角度={np.degrees(angle):.1f}°, 速度={speed}")
    
    print(f"第一阶段最优得分: {best_score:.4f}s")
    
    # 第二阶段：差分进化精细优化
    print("第二阶段：差分进化优化...")
    
    result = differential_evolution(
        objective,
        bounds,
        popsize=25,
        maxiter=80,
        disp=True,
        workers=-1,
        x0=best_params if best_params else None,
        seed=42
    )
    
    final_params = result.x
    final_score = -result.fun
    
    print(f"最终优化得分: {final_score:.6f}s")
    return final_params, final_score

def generate_result_table(optimal_params):
    """生成结果表格"""
    dir_angle, speed = optimal_params[:2]
    drop_delays = optimal_params[2:5]
    det_delays = optimal_params[5:8]
    
    # 方向向量
    dir_xy = np.array([np.cos(dir_angle), np.sin(dir_angle)])
    uav_dir = np.array([dir_xy[0], dir_xy[1], 0.0])
    
    results = []
    
    for i in range(3):
        # 计算投放点和起爆点
        drop_point = calc_drop_point(
            fy1_param["init_pos"], uav_dir, speed, drop_delays[i]
        )
        det_point = calc_det_point(
            drop_point, uav_dir, speed, det_delays[i], g
        )
        
        # 计算单枚弹的有效时间
        single_params = [dir_angle, speed, drop_delays[i], det_delays[i]]
        single_time = -evaluate_shielding_precise(single_params)
        
        results.append({
            '无人机运动方向': np.degrees(dir_angle),
            '无人机运动速度 (m/s)': speed,
            '烟幕干扰弹编号': i + 1,
            '烟幕干扰弹投放点的x坐标 (m)': drop_point[0],
            '烟幕干扰弹投放点的y坐标 (m)': drop_point[1],
            '烟幕干扰弹投放点的z坐标 (m)': drop_point[2],
            '烟幕干扰弹起爆点的x坐标 (m)': det_point[0],
            '烟幕干扰弹起爆点的y坐标 (m)': det_point[1],
            '烟幕干扰弹起爆点的z坐标 (m)': det_point[2],
            '有效干扰时长 (s)': single_time
        })
    
    return results

def solve_problem3():
    """求解问题3"""
    print("开始求解问题3：3枚烟幕弹优化投放策略")
    
    start_time = time.time()
    
    # 优化3枚弹的投放策略
    optimal_params, total_score = optimize_three_missiles()
    
    # 生成结果表格
    results = generate_result_table(optimal_params)
    
    # 输出结果
    print("\n" + "="*80)
    print("问题3优化结果")
    print("="*80)
    print(f"最优飞行角度: {np.degrees(optimal_params[0]):.3f}°")
    print(f"最优飞行速度: {optimal_params[1]:.3f} m/s")
    print(f"投放延迟: {optimal_params[2:5]}")
    print(f"起爆延迟: {optimal_params[5:8]}")
    print(f"总有效遮蔽时间: {total_score:.6f} 秒")
    print("="*80)
    
    # 创建DataFrame并保存
    df = pd.DataFrame(results)
    
    # 保存到Excel文件
    with pd.ExcelWriter('result1.xlsx', engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='问题3结果', index=False)
    
    print("\n结果已保存到 result1.xlsx")
    
    # 详细输出每枚弹的信息
    print("\n=== 各枚烟幕弹详细信息 ===")
    for i, result in enumerate(results, 1):
        print(f"第{i}枚烟幕弹:")
        print(f"  投放点: ({result['烟幕干扰弹投放点的x坐标 (m)']:.3f}, "
              f"{result['烟幕干扰弹投放点的y坐标 (m)']:.3f}, "
              f"{result['烟幕干扰弹投放点的z坐标 (m)']:.3f})")
        print(f"  起爆点: ({result['烟幕干扰弹起爆点的x坐标 (m)']:.3f}, "
              f"{result['烟幕干扰弹起爆点的y坐标 (m)']:.3f}, "
              f"{result['烟幕干扰弹起爆点的z坐标 (m)']:.3f})")
        print(f"  单独有效时长: {result['有效干扰时长 (s)']:.6f} 秒")
        print()
    
    print(f"优化耗时: {time.time() - start_time:.2f} 秒")
    return results, total_score

# 在2.py的主函数后添加问题3的求解
if __name__ == "__main__":
    # 先运行2.py的原有内容...
    
    # 然后求解问题3
    print("\n" + "="*100)
    problem3_results, problem3_score = solve_problem3()