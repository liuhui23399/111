import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time
import copy

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

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

# -------------------------- 2. 核心计算函数 --------------------------
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

def generate_high_density_samples(target, num_circle=80, num_height=25):
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
    radii = np.linspace(0, r, 6, endpoint=True)
    inner_heights = np.linspace(min_z, max_z, 15, endpoint=True)
    inner_thetas = np.linspace(0, 2*np.pi, 20, endpoint=False)
    
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
    """判定真目标是否被多个烟幕的组合效果完全遮蔽"""
    if not smoke_centers:
        return False
    
    # 对每个目标采样点，检查是否被至少一个烟幕遮蔽
    for sample_point in target_samples:
        sample_shielded = False
        
        # 检查当前采样点是否被任何一个烟幕遮蔽
        for smoke_center in smoke_centers:
            if is_segment_intersect_sphere(missile_pos, sample_point, smoke_center, smoke_r):
                sample_shielded = True
                break
        
        # 如果有任何一个采样点没有被任何烟幕遮蔽，则目标未被完全遮蔽
        if not sample_shielded:
            return False
    
    return True

def calc_total_shield_time_multi_uav(params, uav_names, target_samples):
    """计算多架无人机的总遮蔽时间"""
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
            
            # 判断是否被遮蔽（使用组合遮蔽逻辑）
            if active_smoke_centers:
                if is_target_shielded_multi_smoke(missile_pos, active_smoke_centers, smoke_param["r"], target_samples):
                    valid_total += dt
        
        return valid_total
        
    except Exception as e:
        print(f"计算错误: {e}")
        return 0.0

# -------------------------- 3. PSO粒子群优化算法 --------------------------
class Particle:
    """粒子类"""
    def __init__(self, dim, bounds):
        self.dim = dim
        self.bounds = bounds
        
        # 初始化位置和速度
        self.position = np.array([
            np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)
        ])
        
        # 速度初始化为范围的一定比例
        self.velocity = np.array([
            np.random.uniform(-0.1*(bounds[i][1]-bounds[i][0]), 0.1*(bounds[i][1]-bounds[i][0])) 
            for i in range(dim)
        ])
        
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.fitness = float('inf')
    
    def update_velocity(self, global_best_position, w=0.7, c1=1.4, c2=1.4):
        """更新粒子速度"""
        r1 = np.random.random(self.dim)
        r2 = np.random.random(self.dim)
        
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        
        self.velocity = w * self.velocity + cognitive + social
        
        # 速度限制
        for i in range(self.dim):
            v_max = 0.2 * (self.bounds[i][1] - self.bounds[i][0])
            self.velocity[i] = np.clip(self.velocity[i], -v_max, v_max)
    
    def update_position(self):
        """更新粒子位置"""
        self.position += self.velocity
        
        # 边界处理
        for i in range(self.dim):
            if self.position[i] < self.bounds[i][0]:
                self.position[i] = self.bounds[i][0]
                self.velocity[i] = 0
            elif self.position[i] > self.bounds[i][1]:
                self.position[i] = self.bounds[i][1]
                self.velocity[i] = 0
    
    def evaluate(self, fitness_func):
        """评估粒子适应度"""
        self.fitness = fitness_func(self.position)
        
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()

class PSO:
    """粒子群优化算法类"""
    def __init__(self, num_particles=50, max_iterations=200, bounds=None):
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.bounds = bounds
        self.dim = len(bounds)
        
        # 初始化粒子群
        self.particles = [Particle(self.dim, bounds) for _ in range(num_particles)]
        
        # 全局最优
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        
        # 收敛历史
        self.fitness_history = []
        self.diversity_history = []
    
    def calculate_diversity(self):
        """计算种群多样性"""
        positions = np.array([p.position for p in self.particles])
        center = np.mean(positions, axis=0)
        diversity = np.mean([np.linalg.norm(pos - center) for pos in positions])
        return diversity
    
    def optimize(self, fitness_func):
        """执行PSO优化"""
        print(f"开始PSO优化...")
        print(f"粒子数量: {self.num_particles}")
        print(f"最大迭代次数: {self.max_iterations}")
        print(f"参数维度: {self.dim}")
        
        start_time = time.time()
        
        # 初始化评估
        for particle in self.particles:
            particle.evaluate(fitness_func)
            
            if particle.fitness < self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()
        
        print(f"初始最优适应度: {self.global_best_fitness:.6f}")
        
        # 主循环
        for iteration in range(self.max_iterations):
            # 自适应惯性权重
            w = 0.9 - 0.4 * iteration / self.max_iterations
            
            # 自适应学习因子
            c1 = 2.0 - 1.5 * iteration / self.max_iterations
            c2 = 0.5 + 1.5 * iteration / self.max_iterations
            
            # 更新所有粒子
            for particle in self.particles:
                particle.update_velocity(self.global_best_position, w, c1, c2)
                particle.update_position()
                particle.evaluate(fitness_func)
                
                # 更新全局最优
                if particle.fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()
            
            # 记录历史
            self.fitness_history.append(self.global_best_fitness)
            diversity = self.calculate_diversity()
            self.diversity_history.append(diversity)
            
            # 输出进度
            if (iteration + 1) % 20 == 0 or iteration == 0:
                elapsed_time = time.time() - start_time
                print(f"迭代 {iteration+1:3d}: 最优适应度 = {self.global_best_fitness:.6f}, "
                      f"多样性 = {diversity:.4f}, 用时 = {elapsed_time:.2f}s")
            
            # 早停条件
            if len(self.fitness_history) > 50:
                recent_improvement = (self.fitness_history[-50] - self.fitness_history[-1])
                if recent_improvement < 1e-6:
                    print(f"收敛检测: 在第{iteration+1}次迭代停止")
                    break
        
        optimization_time = time.time() - start_time
        print(f"\nPSO优化完成!")
        print(f"最优适应度: {self.global_best_fitness:.6f}")
        print(f"总优化时间: {optimization_time:.2f}秒")
        print(f"实际迭代次数: {len(self.fitness_history)}")
        
        return self.global_best_position, self.global_best_fitness

def pso_optimize_multi_uav(uav_names, target_samples):
    """使用PSO优化多架无人机参数"""
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
    
    def fitness_func(params):
        """适应度函数（最小化问题）"""
        shield_time = calc_total_shield_time_multi_uav(params, uav_names, target_samples)
        return -shield_time  # 转换为最小化问题
    
    # 创建PSO优化器
    pso = PSO(
        num_particles=60,    # 增加粒子数量
        max_iterations=300,  # 增加迭代次数
        bounds=bounds
    )
    
    # 执行优化
    best_position, best_fitness = pso.optimize(fitness_func)
    
    # 创建结果对象（模拟scipy.optimize的返回格式）
    class PSO_Result:
        def __init__(self, x, fun, nfev, success=True):
            self.x = x
            self.fun = fun
            self.nfev = nfev
            self.success = success
    
    # 计算函数评估次数
    nfev = len(pso.fitness_history) * pso.num_particles
    
    result = PSO_Result(best_position, best_fitness, nfev)
    
    # 绘制收敛曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot([-f for f in pso.fitness_history], 'b-', linewidth=2)
    plt.xlabel('迭代次数')
    plt.ylabel('遮蔽时间 (秒)')
    plt.title('PSO收敛曲线')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(pso.diversity_history, 'r-', linewidth=2)
    plt.xlabel('迭代次数')
    plt.ylabel('种群多样性')
    plt.title('种群多样性变化')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pso_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return result

# -------------------------- 4. 主程序 --------------------------
if __name__ == "__main__":
    print("=== 多无人机协同烟幕干扰优化系统 (PSO版本) ===")
    
    # 生成目标采样点
    print("\n生成真目标采样点...")
    target_samples = generate_high_density_samples(real_target, num_circle=80, num_height=25)
    print(f"生成真目标采样点：{len(target_samples)}个")
    
    # 选择参与任务的无人机
    uav_names = ["FY1", "FY2", "FY3"]
    
    print(f"\n开始优化{len(uav_names)}架无人机的协同干扰策略...")
    print("参与无人机：", uav_names)
    for name in uav_names:
        print(f"{name}: {uav_positions[name]}")
    
    print(f"\n目标导弹: M1 位置 {missile_m1['init_pos']}, 速度 {missile_m1['speed']} m/s")
    
    # 执行PSO优化
    start_total = time.time()
    result = pso_optimize_multi_uav(uav_names, target_samples)
    total_time = time.time() - start_total
    
    if result.success:
        print(f"\n=== PSO优化成功！===")
        print(f"最大遮蔽时间：{-result.fun:.6f} 秒")
        print(f"总优化时间：{total_time:.2f} 秒")
        print(f"函数评估次数：{result.nfev}")
        print(f"平均每次评估：{total_time/result.nfev:.4f} 秒")
        
        # 解析最优参数
        optimal_params = result.x
        results_data = []
        
        print(f"\n=== 详细优化结果 ===")
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
            
            # 计算单架无人机的贡献时间
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
            
            print(f"\n{uav_name} PSO优化结果：")
            print(f"  飞行角度：{angle:.6f}°")
            print(f"  飞行速度：{speed:.6f} m/s")
            print(f"  投放延迟：{drop_delay:.6f} s")
            print(f"  起爆延迟：{det_delay:.6f} s")
            print(f"  投放点：{drop_point.round(4)}")
            print(f"  起爆点：{det_point.round(4)}")
            print(f"  单独贡献时间：{single_shield_time:.4f} s")
        
        # 保存到Excel文件
        try:
            df = pd.DataFrame(results_data)
            output_file = "result2_pso.xlsx"
            df.to_excel(output_file, index=False)
            print(f"\n结果已保存到 {output_file}")
        except Exception as e:
            print(f"\n保存Excel失败: {e}")
            try:
                df.to_csv("result2_pso.csv", index=False)
                print("已保存为CSV格式: result2_pso.csv")
            except Exception as e2:
                print(f"CSV保存也失败: {e2}")
        
        print(f"\n*** PSO协同总遮蔽时间：{-result.fun:.6f} 秒 ***")
        
        # 分析协同效果
        individual_sum = sum(float(data["有效干扰时长 (s)"]) for data in results_data)
        cooperative_time = -result.fun
        print(f"\n=== 协同效果分析 ===")
        print(f"个体贡献总和：{individual_sum:.4f} 秒")
        print(f"协同总效果：{cooperative_time:.4f} 秒")
        if cooperative_time > individual_sum:
            synergy = cooperative_time - individual_sum
            print(f"协同增益：{synergy:.4f} 秒 ({(synergy/individual_sum)*100:.1f}%)")
        else:
            print("注意：协同效果分析")
        
    else:
        print("PSO优化失败")

    print(f"\n程序执行完毕，总用时: {total_time:.2f} 秒")