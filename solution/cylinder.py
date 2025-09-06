import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 创建图形和3D坐标轴
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 设置圆柱体参数
radius = 1.0
height = 2.0
resolution = 100  # 圆的解析度

# 创建圆柱体的侧面
x = np.linspace(-radius, radius, resolution)
z = np.linspace(0, height, resolution)
X, Z = np.meshgrid(x, z)
Y = np.sqrt(radius**2 - X**2)

# 绘制圆柱体侧面
ax.plot_surface(X, Y, Z, alpha=0.2, color='lightblue', linewidth=0)
ax.plot_surface(X, -Y, Z, alpha=0.2, color='lightblue', linewidth=0)

# 绘制圆柱体底面
bottom = Circle((0, 0), radius, fill=False, color='black', linewidth=2)
ax.add_patch(bottom)
art3d.pathpatch_2d_to_3d(bottom, z=0, zdir='z')

# 绘制圆柱体顶面
top = Circle((0, 0), radius, fill=False, color='black', linewidth=2)
ax.add_patch(top)
art3d.pathpatch_2d_to_3d(top, z=height, zdir='z')

# 定义特征点
# 顶面上的三个点（间隔90°）
theta = np.linspace(0.5*np.pi, 2*np.pi, 3, endpoint=False)
x_top = radius * np.cos(theta)
y_top = radius * np.sin(theta)
z_top = np.full_like(x_top, height)

# 底面上的三个点（间隔90°）
theta = np.linspace(1.5*np.pi, 3*np.pi, 3, endpoint=False)
x_bottom = radius * np.cos(theta)
y_bottom = radius * np.sin(theta)
z_bottom = np.full_like(x_top, 0)

# 合并所有特征点
x_points = np.concatenate([x_top, x_bottom])
y_points = np.concatenate([y_top, y_bottom])
z_points = np.concatenate([z_top, z_bottom])

# 绘制特征点（红色）
ax.scatter(x_points, y_points, z_points, color='red', s=100)

# 设置坐标轴
ax.set_xlabel('X', fontsize=12, labelpad=10)
ax.set_ylabel('Y', fontsize=12, labelpad=10)
ax.set_zlabel('Z', fontsize=12, labelpad=10)

# 设置坐标轴范围
ax.set_xlim(-radius*1.5, radius*1.5)
ax.set_ylim(-radius*1.5, radius*1.5)
ax.set_zlim(0, height*1.2)

# 确保坐标轴从同一点出发
ax.set_box_aspect([1, 1, 1])  # 设置坐标轴比例
ax.set_proj_type('ortho')     # 使用正交投影

# 设置视角
ax.view_init(elev=20, azim=30)

# 添加标题
plt.title('圆柱体特征点示意图', fontsize=16, pad=20)

# 移除网格线
ax.grid(False)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()