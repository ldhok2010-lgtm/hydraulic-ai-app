import deepxde as dde
import numpy as np

# --- 第一部分：定义物理域 ---
# 假设我们研究的压力范围是 0 到 31.5 MPa
geom = dde.geometry.Interval(0, 31.5)

# --- 第二部分：定义物理规律 (PINN 的核心) ---
# 简化版：内泄露量 Ql = K * P (压差越大，泄露越快)
# 我们让 AI 学习这个斜率 K
def leakage_pde(x, y):
    """
    x: 压力 P
    y: 泄露量 Ql
    """
    dy_dx = dde.grad.jacobian(y, x)
    K = 0.05  # 这是一个假设的泄露系数，实际可以根据油温调整
    return dy_dx - K

# --- 第三部分：提供边界条件 ---
# 压力为 0 时，泄露量理论上为 0
def boundary(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

# --- 第四部分：数据集成 ---
# 模拟几个实测点：压力为 10, 20, 30 MPa 时的泄露量数据
obs_x = np.array([[10], [20], [30]]).astype(np.float32)
obs_y = np.array([[0.5], [1.1], [1.55]]).astype(np.float32)
observe_points = dde.icbc.PointSetBC(obs_x, obs_y, component=0)

data = dde.data.PDE(geom, leakage_pde, [bc, observe_points], num_domain=100, num_boundary=2)

# --- 第五部分：构建并训练 AI ---
net = dde.nn.FNN([1] + [20] * 3 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)
model.compile("adam", lr=0.001)

print("AI 正在学习液压规律，请稍候...")
model.train(iterations=2000)

# --- 第六部分：预测未来 ---
test_pressure = np.array([[25.0]]) # 预测 25MPa 时的泄露情况
prediction = model.predict(test_pressure)
print(f"当压力为 25MPa 时，AI 预测的内泄露量为: {prediction[0][0]:.4f} L/min")

import matplotlib.pyplot as plt

# --- 第七部分：可视化绘图 ---
# 1. 生成 0 到 31.5 MPa 的 100 个压力点用于画线
x_plot = np.linspace(0, 31.5, 100).reshape(-1, 1)
y_plot = model.predict(x_plot)

# 2. 创建画布
plt.figure(figsize=(10, 6))

# 3. 画出 AI 预测的连续曲线（蓝色实线）
plt.plot(x_plot, y_plot, 'b-', label='AI Predicted (Physics-Informed)', linewidth=2)

# 4. 画出你之前提供的 3 个原始实测点（红色圆点）
# 这里的 obs_x 和 obs_y 是你代码里定义过的数据
plt.scatter(obs_x, obs_y, color='red', label='Measured Points', zorder=5)

# 5. 图表修饰（让它看起来专业）
plt.title('Hydraulic Cylinder Internal Leakage Analysis', fontsize=14)
plt.xlabel('Pressure (MPa)', fontsize=12)
plt.ylabel('Leakage Rate (L/min)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 6. 保存并显示
plt.savefig('leakage_curve.png', dpi=300) # 保存高分辨率图片
print("曲线图已生成并保存为 'leakage_curve.png'！")
plt.show()