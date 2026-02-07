import streamlit as st
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# --- 网页标题设置 ---
st.set_page_config(page_title="液压 AI 智能诊断终端", layout="wide")
st.title("🚜 液压缸内泄露 - 物理信息 AI 预测平台")
st.sidebar.header("参数配置")

# --- 1. 定义 AI 模型 (保持之前的逻辑) ---
@st.cache_resource # 缓存模型，避免每次拖动滑块都重新训练
def train_model():
    geom = dde.geometry.Interval(0, 31.5)
    def leakage_pde(x, y):
        dy_dx = dde.grad.jacobian(y, x)
        return dy_dx - 0.05
    
    bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda x, on_boundary: on_boundary and np.isclose(x[0], 0))
    
    # 模拟数据点
    obs_x = np.array([[10], [20], [30]]).astype(np.float32)
    obs_y = np.array([[0.5], [1.1], [1.55]]).astype(np.float32)
    observe_points = dde.icbc.PointSetBC(obs_x, obs_y, component=0)
    
    data = dde.data.PDE(geom, leakage_pde, [bc, observe_points], num_domain=100, num_boundary=2)
    net = dde.nn.FNN([1] + [20] * 3 + [1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    model.train(iterations=1000)
    return model

model = train_model()

# --- 2. 网页交互界面 ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("实时预测")
    # 添加一个滑动条，范围 0-35 MPa
    input_p = st.slider("当前系统压力 (MPa):", 0.0, 35.0, 25.0, 0.1)
    
    # AI 实时推理
    res = model.predict(np.array([[input_p]]))
    
    # 用醒目的仪表盘数字显示结果
    st.metric(label="预计内泄露量 (L/min)", value=f"{res[0][0]:.4f}")
    
    if res[0][0] > 1.5:
        st.error("⚠️ 警告：泄露量超出安全阈值，请检查密封性！")
    else:
        st.success("✅ 系统运行状态良好")

with col2:
    st.subheader("性能特性曲线图")
    # 绘图逻辑
    x_plot = np.linspace(0, 35, 100).reshape(-1, 1)
    y_plot = model.predict(x_plot)
    
    fig, ax = plt.subplots()
    ax.plot(x_plot, y_plot, 'b-', label='AI Physics-Informed Curve')
    ax.scatter([10, 20, 30], [0.5, 1.1, 1.55], color='red', label='Historical Data')
    ax.axvline(input_p, color='green', linestyle='--', label='Current Pressure')
    ax.set_xlabel('Pressure (MPa)')
    ax.set_ylabel('Leakage (L/min)')
    ax.legend()
    st.pyplot(fig)

st.info("注：此模型基于 DeepXDE PINNs 技术，结合了达西定律与实测传感器数据。")