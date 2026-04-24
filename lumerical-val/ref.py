import sys
import numpy as np
import torch

# 1) 导入 lumapi
sys.path.append(r"C:\Program Files\Ansys Inc\v251\Lumerical\api\python")
import lumapi


# -----------------------------
# 2) 读取你训练好的模型
# -----------------------------
ckpt = torch.load("best_model.pth", map_location="cpu")

# 这里要按你自己的 checkpoint 结构改 key
phase_masks = ckpt["phase_mask"].detach().cpu().numpy()   # [num_layers, H, W]
detector_pos = ckpt.get("detector_pos", None)
detector_mask = ckpt.get("detector_mask", None)
detector_minus = ckpt.get("detector_minus", None)

# 你的 config.json 里这些参数也要读出来
wavelength = 532e-9
pixel_size = 8e-6
distance_between_layers = 0.05      # 示例，自己替换
distance_to_detectors = 0.10        # 示例，自己替换

num_layers, H, W = phase_masks.shape

# 相位换成弧度
phase_rads = 2 * np.pi * phase_masks

# 物理坐标
x = (np.arange(W) - W/2) * pixel_size
y = (np.arange(H) - H/2) * pixel_size
z = np.array([0.0])

# -----------------------------
# 3) 构造一个输入场（示例）
#    你这里要替换成测试图像对应的入射场
# -----------------------------
Ex = np.ones((W, H, 1), dtype=np.complex128)   # 注意维度顺序后面要自己核对
Ey = np.zeros((W, H, 1), dtype=np.complex128)
Ez = np.zeros((W, H, 1), dtype=np.complex128)

# 如果你想把图像振幅带进去，可以改成：
# amp = your_image_amplitude  # shape [H, W]
# Ex = amp.T[:, :, None].astype(np.complex128)

# -----------------------------
# 4) 打开 FDTD 会话
# -----------------------------
with lumapi.FDTD(
    hide=True,
    serverArgs={"platform": "offscreen", "threads": "4"}
) as fdtd:

    # 清空工程
    fdtd.eval("switchtolayout; deleteall;")

    # 4.1 建仿真区域（这里只是骨架，span/z 范围你要按系统尺寸改）
    fdtd.addfdtd()
    fdtd.set("x", 0)
    fdtd.set("y", 0)
    fdtd.set("z", 0)
    fdtd.set("x span", W * pixel_size)
    fdtd.set("y span", H * pixel_size)
    fdtd.set("z span", num_layers * distance_between_layers + distance_to_detectors + 20e-6)

    # 4.2 加 imported source
    fdtd.addimportedsource()
    fdtd.set("name", "src")

    # 4.3 把坐标和场送进 Lumerical workspace
    fdtd.putv("x_arr", x)
    fdtd.putv("y_arr", y)
    fdtd.putv("z_arr", z)
    fdtd.putv("Ex_arr", Ex)
    fdtd.putv("Ey_arr", Ey)
    fdtd.putv("Ez_arr", Ez)

    # 4.4 在 Lumerical 里构造 dataset，并导入 source
    fdtd.eval(f"""
    EM = rectilineardataset("EM fields", x_arr, y_arr, z_arr);
    EM.addattribute("E", Ex_arr, Ey_arr, Ez_arr);
    select("src");
    importdataset(EM);
    """)

    # 4.5 这里开始是“你的相位层怎么建”的分叉点
    # ------------------------------------------
    # 方案 A：你已经有真实微结构映射 -> 建几何/材料
    # 方案 B：你只是先做系统级一致性验证 -> 更建议先只验证输入场、传播和探测面结果
    #
    # 如果你后面告诉我是想：
    # 1) 做理想相位屏等效验证
    # 2) 做真实超表面单元 FDTD
    # 我会给你不同版本脚本

    # 示例：添加一个探测 monitor
    fdtd.addprofile()
    fdtd.set("name", "detector_monitor")
    fdtd.set("monitor type", "2D Z-normal")
    fdtd.set("x", 0)
    fdtd.set("y", 0)
    fdtd.set("z", num_layers * distance_between_layers + distance_to_detectors)
    fdtd.set("x span", W * pixel_size)
    fdtd.set("y span", H * pixel_size)

    # 保存并运行
    fdtd.save("onn_check.fsp")
    fdtd.run()

    # 取结果
    E_det = fdtd.getresult("detector_monitor", "E")

# Python 侧做后处理：
# 1) |E|^2
# 2) 按 detector_pos / detector_size 做区域积分
# 3) 乘 detector_mask、减 detector_minus
# 4) 和 evaluate.py / quick_eval.py 对比