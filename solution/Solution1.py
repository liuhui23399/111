import matplotlib.pyplot as plt
import functions

# 图例中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 模型参数
params = [
    180.0, # 角度，向x负方向
    120.0, # m/s 无人机速度
    1.5, # s 投弹延迟
    3.6, # s 起爆延迟
    0.01, # 精确度，通过步长控制
]
uav_initial_pos = functions.UAV_POSITIONS['FY1']
missile_initial_pos = functions.MISSILE_POSITIONS['M1']

# 计算遮蔽时间
total_mask_time, mask_interval, max_distances_to_los, times = functions.calculate_obscuration_time(params, uav_initial_pos, missile_initial_pos)
total_mask_time = -total_mask_time

# 输出结果
print(f"总有效遮蔽时间 = {total_mask_time:.3f} s")
t_start, t_end = mask_interval
print(f"遮蔽开始时间：{t_start:.3f} s, 遮蔽结束时间{t_end:.3f} s")

# 距离-时间图
plt.figure(figsize=(10, 5))
plt.plot(times, max_distances_to_los, label="云团到6条视线中的最远距离")
plt.axhline(functions.cloud_effective_radius, color="r", linestyle="--", label="10 m 遮蔽阈值")
plt.xlabel("时间 (s)")
plt.ylabel("距离 (m)")
plt.title("“全部遮挡”判断：云团到视线锥的最大距离随时间变化")
plt.legend()
plt.grid(True)
plt.xlim(5.1, 10)
plt.ylim(0, 200)
plt.ylim(bottom=0)
plt.show()

