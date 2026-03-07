# %%
import importlib
import numpy as np
import matplotlib.pyplot as plt
import itertools
import copy
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms

# 防止内核挂掉
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# %% [markdown]
# # Angular Spectrum Propagation (phase & amplitude)
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using Device: ',device)
# %% [markdown]
# ### ***Loading and Viewing Dataset***
# %%
BATCH_SIZE = 200
IMG_SIZE = 128          # 合成数据就是 128x128
N_pixels = 128
PADDING = 0             # 已经是 128x128，不需要 pad

# 数据预处理并加载（灰度）
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])
# 把原来的 ImageFolder 创建替换为下面（4->3 映射）
# 原始 mapping: 0:'fundamental',1:'two_lobes',2:'four_lobes',3:'multi_lobes'
new_label_map = {0: 0, 1: 1, 2: 2, 3: 2}  # 把 2 和 3 合并为新的类 2 (high_order)

def target_map(old_label):
    return new_label_map[int(old_label)]

train_dataset = torchvision.datasets.ImageFolder("./data/vcsel_synth/train",
                                                 transform=transform,
                                                 target_transform=target_map)
val_dataset = torchvision.datasets.ImageFolder("./data/vcsel_synth/val",
                                               transform=transform,
                                               target_transform=target_map)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 更新类别数与类名
num_classes = 3
class_names = ['fundamental', 'two_lobes', 'high_order']
print("Using remapped classes:", class_names)
num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)
print("train classes:", train_dataset.class_to_idx)
print("val classes:", val_dataset.class_to_idx)
print("train transform:", train_dataset.transform)
print("val transform:", val_dataset.transform)
# 定义一个绘图函数
def image_plot(image, label):
#     cmap='RdBu'
    fig, ax = plt.subplots()
    ax.imshow(np.round(image.cpu().numpy(), 5)) # 显示图片每个像素点的振幅
    ax.axis('off')
    ax.set_title(label.cpu().numpy())
#     fig.colorbar(plt.cm.ScalarMappable(cmap=cmap))
    plt.show()

plt.rcParams["figure.figsize"] = (20, 6)
for i, (images, labels) in enumerate(train_dataloader):
    images = images.to(device)
    images_E = torch.sqrt(torch.squeeze(images))
    labels = labels.to(device)

    # 每100个批次绘制第一张图片
    if (i + 1) % 100 == 0:
        classes = torch.unique(labels).cpu().numpy()
        classes_num = len(classes)
        print('batch_number [{}/{}]'.format(i + 1, len(train_dataloader)))
        print('classes of the first batch: {}, number of classes: {}'.format(classes, classes_num))
        image_plot(images_E[0], labels[0])
# %%
images_E.shape
# %% [markdown]
# ### ***Diffractive Layer***
# %%
class Diffractive_Layer(torch.nn.Module):
    # 模型初始化（构造实例），默认实参波长为830e-9，网格总数50，网格大小2e-6，z方向传播0.002。
    def __init__(self, λ = 830e-9, N_pixels = 128, pixel_size = 2e-6, distance = torch.tensor([0.005])):
        super(Diffractive_Layer, self).__init__() # 初始化父类
        
        # 以1/d为单位频率，得到一系列频率分量[0, 1, 2, ···, N_pixels/2-1,-N_pixels/2, ···, -1]/(N_pixels*d)。
        fx = np.fft.fftshift(np.fft.fftfreq(N_pixels, d = pixel_size))
        fy = np.fft.fftshift(np.fft.fftfreq(N_pixels, d = pixel_size))
        fxx, fyy = np.meshgrid(fx, fy) # 拉网格，每个网格坐标点为空间频率各分量。

        argument = (2 * np.pi)**2 * ((1. / λ) ** 2 - fxx ** 2 - fyy ** 2)

        # 计算传播场或倏逝场的模式kz，传播场kz为实数，倏逝场kz为复数
        tmp = np.sqrt(np.abs(argument))
        self.distance = distance.to(device)
        self.kz = torch.tensor(np.where(argument >= 0, tmp, 1j*tmp)).to(device)

    def forward(self, E):
        # 定义单个衍射层内的前向传播
        fft_c = torch.fft.fft2(E) # 对电场E进行二维傅里叶变换
        c = torch.fft.fftshift(fft_c).to(device) # 将零频移至张量中心
        phase = torch.exp(1j * self.kz * self.distance).to(device)
        angular_spectrum = torch.fft.ifft2(torch.fft.ifftshift(c * phase)) # 卷积后逆变换得到响应的角谱
        return angular_spectrum
# %% [markdown]
# ### ***Propagation Layer***
# %%
class Propagation_Layer(torch.nn.Module):
    # 与上面衍射层大致相同，区别在于传输层是最后一个衍射层到探测器层间的部分，中间可以自定义加额外的器件。
    def __init__(self, λ = 830e-9, N_pixels = 128, pixel_size = 2e-6, distance = torch.tensor([0.001])):
        super(Propagation_Layer, self).__init__() # 初始化父类
        
        # 以1/d为单位频率，得到一系列频率分量[0, 1, 2, ···, N_pixels/2-1,-N_pixels/2, ···, -1]/(N_pixels*d)。
        fx = np.fft.fftshift(np.fft.fftfreq(N_pixels, d = pixel_size))
        fy = np.fft.fftshift(np.fft.fftfreq(N_pixels, d = pixel_size))
        fxx, fyy = np.meshgrid(fx, fy) # 拉网格，每个网格坐标点为空间频率各分量。

        argument = (2 * np.pi)**2 * ((1. / λ) ** 2 - fxx ** 2 - fyy ** 2)

        # 计算传播场或倏逝场的模式kz，传播场kz为实数，倏逝场kz为复数
        tmp = np.sqrt(np.abs(argument))
        self.distance = distance.to(device)
        self.kz = torch.tensor(np.where(argument >= 0, tmp, 1j*tmp)).to(device)

    def forward(self, E):
        # 定义单个衍射层内的前向传播
        fft_c = torch.fft.fft2(E) # 对电场E进行二维傅里叶变换
        c = torch.fft.fftshift(fft_c) # 将零频移至张量中心
        phase = torch.exp(1j * self.kz * self.distance).to(device)
        angular_spectrum = torch.fft.ifft2(torch.fft.ifftshift(c * phase)) # 卷积后逆变换得到响应的角谱
        return angular_spectrum
# %%
def propagation_along_z(initial_field, wl, N_pixels, pixel_size, z_step, number_of_z_steps):
    diffraction_step = Diffractive_Layer(distance=torch.tensor(z_step), λ=wl, N_pixels=N_pixels, pixel_size=pixel_size)
    z_ind = np.arange(0, number_of_z_steps)
    full_cross_section = torch.zeros((len(z_ind), N_pixels, N_pixels), dtype=torch.complex64)
    full_cross_section[0] = initial_field.detach().clone()
    with torch.no_grad():
        for ind in z_ind[1:]:
            full_cross_section[ind] = diffraction_step(full_cross_section[ind-1])
        return full_cross_section

DISCRETIZATION_STEP = 2.0e-6 # 网格大小
N_pixels = 128 # xy平面边长的元素点数量
z_step = 1e-5 # z方向步长
number_of_z_steps = 500 # z方向步数
z = np.arange(0, number_of_z_steps)*z_step # z坐标
wl = 830e-9 # 波长
coord_limit = (N_pixels//2)*DISCRETIZATION_STEP # 建立xy坐标系
# 拉网格
mesh = np.arange(-coord_limit, coord_limit, DISCRETIZATION_STEP)
x, y = np.meshgrid(mesh, mesh)
# %%
images_input = torch.flip(images_E[0], [0])
field = propagation_along_z(images_input, wl, N_pixels, DISCRETIZATION_STEP, z_step, number_of_z_steps)

plt.rcParams["figure.figsize"] = (20, 6)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout=True)
cmap = 'seismic'

# input
XY_field_in = np.abs(field[0].data.numpy())
ax1.pcolormesh(mesh*10**6, mesh*10**6, XY_field_in, cmap=cmap)
ax1.set_xlabel('x ($\mu$m)')
ax1.set_ylabel('y ($\mu$m)')

# yz cross-section
YZ_field = torch.abs(field[:, :, N_pixels//2])
ax2.pcolormesh(z*10**3, mesh*10**6, YZ_field.T, cmap=cmap)
ax2.set_xlabel('z (mm)')
ax2.set_ylabel('y ($\mu$m)')

# output
XY_field_out = torch.abs(field[-1]).T
ax3.pcolormesh(mesh*10**6, mesh*10**6, XY_field_out.T, cmap=cmap)
ax3.set_xlabel('x ($\mu$m)')
ax3.set_ylabel('y ($\mu$m)')

plt.show()
# %%
class Lens_Layer(torch.nn.Module):
    # 模型初始化（构造实例），默认实参波长为830e-9，网格总数50，网格大小20e-6，z方向传播0.002。
    def __init__(self, f = 5e-3, wl = 830e-9, N_pixels = 128, pixel_size = 2e-6, distance = torch.tensor([0.005])):
        super(Lens_Layer, self).__init__() # 初始化父类
        # 网格大小
        coord_limit = (N_pixels//2)*pixel_size # 建立xy坐标系
        # 拉网格
        mesh = np.arange(-coord_limit, coord_limit, pixel_size)
        x, y = np.meshgrid(mesh, mesh)
        self.phase_lens = torch.tensor(np.exp(-1j*np.pi/(wl*f) * (x**2 + y**2))).to(device)
        
        # 以1/d为单位频率，得到一系列频率分量[0, 1, 2, ···, N_pixels/2-1,-N_pixels/2, ···, -1]/(N_pixels*d)。
        fx = np.fft.fftshift(np.fft.fftfreq(N_pixels, d = pixel_size))
        fy = np.fft.fftshift(np.fft.fftfreq(N_pixels, d = pixel_size))
        fxx, fyy = np.meshgrid(fx, fy) # 拉网格，每个网格坐标点为空间频率各分量。

        argument = (2 * np.pi)**2 * ((1. / wl) ** 2 - fxx ** 2 - fyy ** 2)

        # 计算传播场或倏逝场的模式kz，传播场kz为实数，倏逝场kz为复数
        tmp = np.sqrt(np.abs(argument))
        self.distance = distance.to(device)
        self.kz = torch.tensor(np.where(argument >= 0, tmp, 1j*tmp)).to(device)

    def forward(self, E):
        # 定义单个衍射层内的前向传播
        fft_c = torch.fft.fft2(E) # 对电场E进行二维傅里叶变换
        c = torch.fft.fftshift(fft_c).to(device) # 将零频移至张量中心
        phase = torch.exp(1j * self.kz * self.distance).to(device)
        angular_spectrum = torch.fft.ifft2(torch.fft.ifftshift(c * phase)) # 卷积后逆变换得到响应的角谱
        return angular_spectrum*self.phase_lens
# %%
# 不看中间的光束传输过程，可以根据角谱直接计算出z位置的光场分布，结果与考虑传输过程相同
images_input = torch.flip(images_E[0], [0])
field_test = Lens_Layer(distance=torch.tensor(5e-3))(images_input)

# output
plt.rcParams["figure.figsize"] = (4, 4)
fig, ax = plt.subplots(1)
XY_field_out = torch.abs(field_test).T.cpu().numpy()
ax.pcolormesh(mesh*10**6, mesh*10**6, XY_field_out.T, cmap=cmap)
ax.set_xlabel('x ($\mu$m)')
ax.set_ylabel('y ($\mu$m)')

plt.show()
# %%
# 在傅里叶平面上设置一个正方形光阑，光阑矩阵
square_aper = torch.zeros((N_pixels,N_pixels), dtype = torch.double)
L_side = int(N_pixels*1) # 1：全透，1/2中央二分之一区域透光
x1 = int((N_pixels-L_side)/2)
x2 = int((N_pixels+L_side)/2)
square_aper[x1:x2, x1:x2] = 1 # 设置探测器区域

f = 5e-3
field_lens1 = field[-1]*np.exp(-1j*np.pi/(wl*f) * (x**2 + y**2))
field_out_lens1_F1 = propagation_along_z(field_lens1, wl, N_pixels, DISCRETIZATION_STEP, z_step, number_of_z_steps)
field_out_lens1_F1[-1] = field_out_lens1_F1[-1] * square_aper
field_out_lens1_F2 = propagation_along_z(field_out_lens1_F1[-1], wl, N_pixels, DISCRETIZATION_STEP, z_step, number_of_z_steps)
field_lens2 = field_out_lens1_F2[-1]*np.exp(-1j*np.pi/(wl*f) * (x**2 + y**2))
field_out_lens2 = propagation_along_z(field_lens2, wl, N_pixels, DISCRETIZATION_STEP, z_step, number_of_z_steps)

YZ_field_lens1_F1 = torch.abs(field_out_lens1_F1[:, N_pixels//2])
YZ_field_lens1_F2 = torch.abs(field_out_lens1_F2[:, N_pixels//2])
YZ_field_lens2 = torch.abs(field_out_lens2[:, N_pixels//2])
YZ_field_total = torch.cat((YZ_field, YZ_field_lens1_F1, YZ_field_lens1_F2, YZ_field_lens2), dim = 0)
z_total = np.arange(0, 4*number_of_z_steps)*z_step # z坐标
# %%
plt.rcParams["figure.figsize"] = (20, 6)
# yx cross-section
fig, ax = plt.subplots(1, constrained_layout=True)
ax.pcolormesh(z_total*10**3, mesh*10**6, YZ_field_total.T, cmap = 'seismic')
ax.set_xlabel('z (mm)')
ax.set_ylabel('y ($\mu$m)')

plt.annotate('', xy=(max(z_total*10**3)/4, -coord_limit*10**6), xytext=(max(z_total*10**3)/4, coord_limit*10**6), 
             arrowprops=dict(arrowstyle='<->, head_length=3, head_width=1', lw=3, color='red'))

plt.annotate('', xy=(max(z_total*10**3)/2, -coord_limit*10**6), xytext=(max(z_total*10**3)/2, coord_limit*10**6), 
             arrowprops=dict(arrowstyle='-', lw=3, color='red'))

plt.annotate('', xy=(max(z_total*10**3)*3/4, -coord_limit*10**6), xytext=(max(z_total*10**3)*3/4, coord_limit*10**6), 
             arrowprops=dict(arrowstyle='<->, head_length=3, head_width=1', lw=3, color='red'))

plt.show()
# %%
plt.rcParams["figure.figsize"] = (10, 10)
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
# input
XY_field_in = np.abs(field[0].data.numpy())
ax1.pcolormesh(mesh*10**6, mesh*10**6, XY_field_in, cmap=cmap)
ax1.set_xlabel('x ($\mu$m)')
ax1.set_ylabel('y ($\mu$m)')
ax1.set_aspect('equal')
ax1.set_title('Int = {}'.format(torch.pow(abs(torch.tensor(XY_field_in)), 2).sum()))

# output
XY_field_out_lens2 = torch.abs(field_out_lens2[-1]).T
ax2.pcolormesh(mesh*10**6, mesh*10**6, XY_field_out_lens2.T, cmap=cmap)
ax2.set_xlabel('x ($\mu$m)')
ax2.set_ylabel('y ($\mu$m)')
ax2.set_aspect('equal')
ax2.set_title('Int = {}'.format(torch.pow(abs(XY_field_out_lens2), 2).sum()))

plt.axis('square')
plt.show()
# %%
# 在傅里叶平面上设置一个正方形光阑，光阑矩阵
square_aper = torch.zeros((N_pixels,N_pixels), dtype = torch.double)
L_side = int(N_pixels*1) # 1：全透，1/2中央二分之一区域透光
x1 = int((N_pixels-L_side)/2)
x2 = int((N_pixels+L_side)/2)
square_aper[x1:x2, x1:x2] = 1 # 设置探测器区域

# 不看中间的光束传输过程，可以根据角谱直接计算出z位置的光场分布，结果与考虑传输过程相同
images_input = torch.flip(images_E[0], [0])
field_test1 = Lens_Layer(distance=torch.tensor(5e-3))(images_input) # 物面到第一个lens
field_test2 = Diffractive_Layer(distance=torch.tensor(5e-3))(field_test1) # 第一个lens到傅里叶平面
field_test3 = field_test2 * square_aper.to(device) # 经过傅里叶平面的光阑
field_test4 = Lens_Layer(distance=torch.tensor(5e-3))(field_test3) # 傅里叶平面到第二个lens
field_test5 = Propagation_Layer(distance=torch.tensor(5e-3))(field_test4) # 第二个lens到像面

# output
plt.rcParams["figure.figsize"] = (4, 4)
fig, ax = plt.subplots(1)
XY_field_out = torch.abs(field_test5).T.cpu().numpy()
ax.pcolormesh(mesh*10**6, mesh*10**6, XY_field_out.T, cmap=cmap)
ax.set_xlabel('x ($\mu$m)')
ax.set_ylabel('y ($\mu$m)')

plt.show()
# %% [markdown]
# ### ***Detectors Layer***
# %%
# 生成一行探测器。指定探测器个数N_det，在x方向上生成齐高等间距det_step的一组探测器
# left，right，up和down分别是该行矩形探测器的四个顶点坐标。
def generate_det_row(det_size, start_pos_x, start_pos_y, det_step, N_det):
    p = []
    for i in range(N_det):
        left = start_pos_x + i * (int(det_step) + det_size)
        right = left + det_size
        up = start_pos_y
        down = start_pos_y + det_size
        p.append((up, down, left, right))
    return p

# 获取最终衍射光强在各个探测器上的分布情况
def detector_region_logits(Int):
    """
    Int: [B, H, W] intensity (float)
    返回: logits [B, C] = 每个探测器区域的总强度（未归一化）
    """
    region_scores = []
    # detector_pos 中保存 (up, down, left, right)
    for up, down, left, right in detector_pos:
        region_sum = Int[:, up:down, left:right].sum(dim=(1, 2))  # [B]
        region_scores.append(region_sum)
    logits = torch.stack(region_scores, dim=1)  # [B, C]
    return logits
# 你训练集类别数
num_classes = len(train_dataset.classes)

# 自动生成一行探测器，保证不越界
# ---------- 重新生成 3 个探测器的布局（替换原有 detector 生成段） ----------
# 确保 num_classes == 3（如果尚未修改，请先设置 num_classes = 3）

# 设计参数（可调）
det_size = 24   # 每个探测器边长（像素），可根据需要调整（例如 16..32）
gap = 8         # 探测器之间的水平间隙（像素）

# 计算并居中放置
total_width = num_classes * det_size + (num_classes - 1) * gap
start_pos_x = (N_pixels - total_width) // 2   # 左边起点（列方向）
start_pos_y = (N_pixels - det_size) // 2      # 上边起点（行方向），竖直居中

# 生成探测器位置列表，generate_det_row 返回 (up, down, left, right)
detector_pos = generate_det_row(det_size, start_pos_x, start_pos_y, gap, num_classes)

# 生成每类对应的 labels_image_tensors（每个 tensor 是该探测器区域为 1，其它为 0，并归一化到总和为 1）
labels_image_tensors = torch.zeros((num_classes, N_pixels, N_pixels), device=device, dtype=torch.float32)
for ind, pos in enumerate(detector_pos):
    up, down, left, right = pos
    # 注意：张量索引 order 是 [rows (y), cols (x)] == [up:down, left:right]
    labels_image_tensors[ind, up:down, left:right] = 1.0
    # 归一化（区域内强度和为 1）
    labels_image_tensors[ind] = labels_image_tensors[ind] / labels_image_tensors[ind].sum()

# 合成一个可视化的 detector layout（二维）
det_ideal = labels_image_tensors.cpu().numpy().sum(axis=0)

# 可视化 probe（在 notebook/脚本里显示一次以确认位置）
plt.figure(figsize=(4,4))
plt.imshow(det_ideal, cmap="gray", origin="upper")
plt.title(f"Detector layout (num_det={num_classes}, det_size={det_size}, gap={gap})")
plt.axis('off')
plt.show()

# 打印 detector_pos 以便调试/确认（每项是 (up,down,left,right)）
print("Detector positions (up, down, left, right):")
for i, p in enumerate(detector_pos):
    print(f"  det[{i}] = {p}")
# -----------------------------------------------------------------------------
# %% [markdown]
# ### ***D2NN***
# %%
# 先做一个模型初始化，将生成的初始随机相位参数与模型分离，以便于后面对其他参数分别训练。
# 初始化每层相位板的相位参数（0到1区间均匀分布）,并将其注册为可学习的Parameter。
num_layers = 3
distance_between_layers = 0.004, 0.002,0.004, 0.004, 0.005
phase = [torch.nn.Parameter(torch.from_numpy(np.random.random(size=(N_pixels, N_pixels)).
                                                          astype('float32'))) for _ in range(num_layers)]
# Amp = [torch.nn.Parameter(torch.from_numpy(np.random.random(size=(N_pixels, N_pixels)).
#                                                           astype('float32'))) for _ in range(num_layers)]
# 初始化层间距，预训练时（先只训练相位）设置梯度requires_grad=False。
distance = [torch.nn.Parameter(distance_between_layers[i]*torch.tensor([1])) for i in range(num_layers+2)]
# %%
distance[1]
# %%
class DNN(torch.nn.Module):
    """""""""""""""""""""
    phase & amplitude modulation
    """""""""""""""""""""
    def __init__(self, phase = [], num_layers = 3, wl = 830e-9, N_pixels = 128, pixel_size = 20e-6,
                 distance = []):

        super(DNN, self).__init__()
        
        # 初始化每层相位板的相位参数（0到1区间均匀分布）,并将其注册为可学习的Parameter。
#         self.phase = [torch.nn.Parameter(torch.from_numpy(np.random.random(size=(N_pixels, N_pixels)).
#                                                           astype('float32')-0.5)) for _ in range(num_layers)]
        # 向网络中添加每层相位板的参数
        for i in range(num_layers):
            self.register_parameter("phase" + "_" + str(i), phase[i])
        # 向网络中添加每层相位板的衰减参数
#         for i in range(num_layers):
#             self.register_parameter("Amp" + "_" + str(i), Amp[i])
        # 向网络中添加层间距的参数
        for i in range(num_layers+2):
            self.register_parameter("distance" + "_" + str(i), distance[i])
        # 定义最大相位
        self.phi_max = np.pi

        # 定义透镜面
        self.lens_layer1 = Lens_Layer()
        
        # 定义中间的衍射层
        self.diffractive_layers = torch.nn.ModuleList([Diffractive_Layer(wl, N_pixels, pixel_size, distance[i])
                                                       for i in range(num_layers)])
        
        # 在相位板后加一层饱和吸收体，用ReLU模拟非线性
#         self.relu = torch.nn.ReLU()
        
        # 定义透镜面
        self.lens_layer2 = Lens_Layer(distance=distance[-2])
        
        # 定义最后的探测层
        self.last_diffractive_layer = Propagation_Layer(wl, N_pixels, pixel_size, distance[-1])
        self.sofmax = torch.nn.Softmax(dim=-1)
    
    # 计算多层衍射前向传播
    def forward(self, E):
        E = self.lens_layer1(E)
        for index, layer in enumerate(self.diffractive_layers):
            temp = layer(E)
            registered_phase = getattr(self, f"phase_{index}")
            constr_phase = np.pi * registered_phase
            exp_j_phase = torch.exp(1j*constr_phase) #torch.cos(constr_phase)+1j*torch.sin(constr_phase)
            E = temp * exp_j_phase
            # 【小修改】添加 eps 防止 |E|≈0 时除零产生 NaN
            E_phase = E / (torch.abs(E) + 1e-12)
            I = torch.abs(E) ** 2  # real float32
            I_th = torch.mean(I, dim=(-2, -1), keepdim=True) / 2.0  # float32
            I_out = torch.nn.functional.relu(I - I_th)  # float32
            E_phase = E / (torch.abs(E) + 1e-12)
            E = torch.sqrt(I_out + 1e-12) * E_phase
        E = self.lens_layer2(E)
        E = self.last_diffractive_layer(E)
        Int = torch.abs(E)**2
#         output = self.sofmax(detector_region(Int))
        output = detector_region_logits(Int)
        return output, Int
# %%
a = torch.tensor([[[-1, 1, 1], [4, -4, 4]], [[2, 2, -2], [5, -5, 5]]]).to(dtype=torch.float64)
print(a.size())
print(a)
b = torch.mean(torch.abs(a), dim=(-2, -1), keepdim=True)
print(b.size())
c = torch.ones_like(a)*b
c.size()
print(c)

a-c
# %%
I = torch.abs(images_E)**2
I_th = torch.mean(I, dim=(-2, -1), keepdim=True)
I_th = I_th * torch.ones_like(images_E)
I_th.shape
I_out = torch.nn.ReLU()(I-I_th)
I_out.shape
# %%
model = DNN(phase = phase, distance = distance).to(device)
print(model)
# %%
# 展示可供训练的参数（因为）
for index, item in model.named_parameters():
    print(index)
# %%
# 先只训练相位，层间距先设定0.005。之后再在0.005这个基础上微调训练层间距。
for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if name.find('phase') != -1:
        exec('model.' + name + '.requires_grad_(True)')
# for name, param in model.named_parameters():
#     if name.find('Amp') != -1:
#         exec('model.' + name + '.requires_grad_(True)')
    
print("Params to learn:")
params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)
# %% [markdown]
# ### ***Training***
# %%
def train(model, loss_function, optimizer, trainloader, testloader, epochs=10, device='cpu', filename='best.pt'):
    """
    Training loop using CrossEntropyLoss.
    Expects model(images) -> (logits, Int) where `logits` are raw class scores [B, C]
    loss_function should be torch.nn.CrossEntropyLoss()
    Returns: train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist, best_model
    """
    train_loss_hist = []
    test_loss_hist = []
    train_acc_hist = []
    test_acc_hist = []
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in tqdm(trainloader):
            # Prepare inputs
            images = torch.sqrt(images.to(device).squeeze())
            labels = labels.to(device).long()

            optimizer.zero_grad()

            # Forward
            out_label, out_img = model(images)  # out_label: logits [B, C]
            # Ensure dtype
            out_label = out_label.to(dtype=torch.float32)

            # Loss (CrossEntropy expects raw logits and Long labels)
            loss = loss_function(out_label, labels)
            loss.backward()
            optimizer.step()

            # Stats
            running_loss += loss.item()
            _, predicted = torch.max(out_label, 1)
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)

        epoch_train_loss = running_loss / len(trainloader)
        epoch_train_acc = running_correct / running_total if running_total > 0 else 0.0
        train_loss_hist.append(epoch_train_loss)
        train_acc_hist.append(epoch_train_acc)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(testloader):
                images = torch.sqrt(images.to(device).squeeze())
                labels = labels.to(device).long()

                out_label, out_img = model(images)
                out_label = out_label.to(dtype=torch.float32)

                loss = loss_function(out_label, labels)
                val_loss += loss.item()

                _, predicted = torch.max(out_label, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / len(testloader)
        epoch_val_acc = val_correct / val_total if val_total > 0 else 0.0
        test_loss_hist.append(epoch_val_loss)
        test_acc_hist.append(epoch_val_acc)

        # Save best model
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            state = {
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, filename)

        print(f"Epoch={epoch} train loss={epoch_train_loss:.4f}, test loss={epoch_val_loss:.4f}")
        print(f"train acc={epoch_train_acc:.4f}, test acc={epoch_val_acc:.4f}")
        print("-----------------------")

    # load best weights
    model.load_state_dict(best_model_wts)
    return train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist, model
# %%
def custom_loss(imgs, det_labels):
    full_int = imgs.sum(dim=(1,2))
    loss = 1 - (imgs*det_labels).sum(dim=(1,2))/full_int
    return loss.mean()
# %%
# 定义衍射模型基本参数
wl = 830e-9
# pixel_size = 10*wl
pixel_size = 2e-6
# 定义模型，损失函数和优化器
model = DNN(phase = phase, num_layers = 3, wl = wl, pixel_size = pixel_size, distance = distance).to(device)
# criterion = custom_loss
criterion = torch.nn.CrossEntropyLoss().to(device)
# criterion = torch.nn.MSELoss(reduction='mean').to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
optimizer = torch.optim.Adam(params_to_update, lr=0.002)
# %%
# 正式开启训练
train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist, best_model = train(model, 
                          criterion,optimizer, train_dataloader, val_dataloader, epochs = 10,  device = device)
# %%
# 查看训练的各个参数，确保这一次只有相位被训练了（requires_grad=True）
for param in model.named_parameters():
    print(param)
# %%
# 全部解禁，相位和层间距都拿来训练
for param in model.parameters():
    param.requires_grad = True

# for name, param in model.named_parameters():
#     if name.find('distance') != -1:
#         exec('model.' + name + '.requires_grad_(True)')

print("Params to learn:")
params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)
# %%
# 加载之前训练好的参数
checkpoint = torch.load('best.pt')
best_acc = checkpoint['best_acc']
model.load_state_dict(checkpoint['state_dict'])
# %%
# 加载损失函数和优化器
# criterion = torch.nn.MSELoss(reduction='sum').to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(params_to_update, lr=1e-6)
# %%
# 重新开启训练
train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist, best_model = train(model, 
                          criterion,optimizer, train_dataloader, val_dataloader, epochs = 10,  device = device)
# %%
for name, param in model.named_parameters():
    print(param)
# %%
# === 插入点：放在训练结束（train(...) 返回并且 model/best_model 可用）之后 ===
# 自动对训练好的 model 进行 4-level / 8-level / 16-level 相位量化并在 val_dataloader 上评估

import copy
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 小工具：检测 phase 参数映射（linear 或 sigmoid）
def detect_phase_mapping(model):
    mins, maxs = [], []
    for name, p in model.named_parameters():
        if name.startswith("phase_"):
            arr = p.detach().cpu().numpy()
            mins.append(float(np.min(arr)))
            maxs.append(float(np.max(arr)))
    if not mins:
        raise RuntimeError("模型中未找到以 'phase_' 开头的参数。")
    min_all, max_all = min(mins), max(maxs)
    # 参数落在 [0,1] 更可能是 linear 映射 (phi = π * p)，否则可能是 sigmoid(param)
    return 'linear' if (min_all >= -1e-6 and max_all <= 1.0 + 1e-6) else 'sigmoid'

# 通用量化（支持 linear 与 sigmoid 两种映射）
def quantize_model_phases_general(model_src, L, mapping='linear', eps=1e-6):
    model_q = copy.deepcopy(model_src)
    device_q = next(model_q.parameters()).device
    levels = (torch.arange(L, dtype=torch.float32) * (np.pi / L))  # phi levels: 0, π/L, ..., (L-1)π/L

    for name, param in model_src.named_parameters():
        if not name.startswith("phase_"):
            continue
        orig = param.detach().cpu().to(torch.float32)
        if mapping == 'linear':
            phi = np.pi * orig
            diffs = torch.abs(phi.unsqueeze(-1) - levels.unsqueeze(0))
            idx = torch.argmin(diffs, dim=-1)
            phi_q = levels[idx]
            p_q = (phi_q / np.pi).to(dtype=param.dtype)
            param_q = dict(model_q.named_parameters())[name]
            param_q.data.copy_(p_q.to(device_q))
        else:  # sigmoid mapping
            sig = torch.sigmoid(orig)
            phi = np.pi * sig
            diffs = torch.abs(phi.unsqueeze(-1) - levels.unsqueeze(0))
            idx = torch.argmin(diffs, dim=-1)
            phi_q = levels[idx]
            target_sig = (phi_q / np.pi).clamp(eps, 1.0 - eps)
            p_q_val = torch.log(target_sig / (1.0 - target_sig))  # inverse sigmoid (logit)
            param_q = dict(model_q.named_parameters())[name]
            param_q.data.copy_(p_q_val.to(device_q).to(dtype=param.dtype))
    return model_q

# 评估函数（返回 accuracy）
def evaluate_accuracy_model(model_in, dataloader, device):
    model_in.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = torch.sqrt(images.to(device).squeeze())
            labels = labels.to(device)
            out = model_in(images)
            if isinstance(out, (tuple, list)):
                out = out[0]
            out = out.to(torch.float32)
            _, predicted = torch.max(out, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0

# ----------------- 主流程 -----------------
# 确保 val_dataloader 与 model 在会话中存在（训练刚完成时通常满足）
if 'val_dataloader' not in globals():
    raise RuntimeError("val_dataloader 未在当前会话中找到。请先运行数据加载单元。")
if 'model' not in globals():
    raise RuntimeError("model 未在当前会话中找到。请先训练或加载模型。")

model.to(device)
model.eval()

mapping = detect_phase_mapping(model)
print("Detected phase mapping:", mapping)

acc_orig = evaluate_accuracy_model(model, val_dataloader, device)
print(f"Original validation accuracy: {acc_orig*100:.2f}%")

# 批量量化评估：4, 8, 16 levels
for L in (4, 8, 16):
    print(f"\nQuantizing to {L} levels...")
    model_q = quantize_model_phases_general(model, L, mapping=mapping)
    model_q.to(device)
    acc_q = evaluate_accuracy_model(model_q, val_dataloader, device)
    print(f"{L}-level quantized validation accuracy: {acc_q*100:.2f}%")
    print(f"Absolute drop: {(acc_orig - acc_q) * 100.0:.2f} percentage points")

# 保存量化后的 state_dict 以便后续分析（可选）
torch.save({
    'orig_state_dict': model.state_dict(),
    'model_4_state_dict': quantize_model_phases_general(model, 4, mapping=mapping).state_dict(),
    'model_8_state_dict': quantize_model_phases_general(model, 8, mapping=mapping).state_dict(),
    'model_16_state_dict': quantize_model_phases_general(model, 16, mapping=mapping).state_dict(),
}, 'quantized_models_4_8_16.pt')

print("Quantization + evaluation done.")
# %%
# def constrain_phase_0_pi(phase_param: torch.Tensor) -> torch.Tensor:
#     """
#     将“可训练相位参数”(任意实数) 映射为物理相位 [0, π]（单位：rad）。
#     训练时 forward 用的也应是同一个口径：π * sigmoid(param)。
#     """
#     return torch.pi * torch.sigmoid(phase_param)
#
# T_OXIDE = 519e-9  # SiO₂ 总膜厚 (m)
# def quantize_phase_4level(phi_rad):
#     """将连续相位量化为 4 级: {0, π/4, π/2, 3π/4}"""
#     levels = torch.tensor([0, np.pi/4, np.pi/2, 3*np.pi/4])
#     diff = torch.abs(phi_rad[..., None] - levels)
#     return levels[torch.argmin(diff, dim=-1)]
#
#     # 计算每个像素到 4 个 level 的距离并取最小
#     # phi_0_pi: [H, W] or [B, H, W]
#     diffs = (phi_0_pi.unsqueeze(-1) - levels).abs()  # [..., 4]
#     idx = diffs.argmin(dim=-1)  # [...]
#     return levels[idx]
#
#
# def phase_to_height_sio2_paper(phi_quantized: torch.Tensor) -> torch.Tensor:
#     """
#     按你给的文献数据，把四级相位映射到 SiO2 蚀刻深度（单位：nm）：
#     0 -> 519 nm
#     π/4 -> 346 nm
#     π/2 -> 173 nm
#     3π/4 -> 0 nm
#
#     注意：这是“离散表”映射，不依赖折射率与波长。
#     """
#     levels = torch.tensor(
#         [0.0, 0.25 * torch.pi, 0.5 * torch.pi, 0.75 * torch.pi],
#         device=phi_quantized.device,
#         dtype=phi_quantized.dtype,
#     )
#     heights_nm = torch.tensor(
#         [519.0, 346.0, 173.0, 0.0],
#         device=phi_quantized.device,
#         dtype=torch.float32,
#     )
#
#     # 因为 phi_quantized 已经是四级之一，直接找匹配的 level
#     # 用距离最小来索引更稳健
#     diffs = (phi_quantized.unsqueeze(-1) - levels).abs()
#     idx = diffs.argmin(dim=-1)
#     return heights_nm[idx]
#
#
# def phase_to_height_physics(phi: torch.Tensor, lambda0=830e-9, n_sio2=1.46, n_env=1.0) -> torch.Tensor:
#     """
#     可选：基于物理公式的连续映射（单位：m）。
#     Δφ = (2π/λ) * (n_sio2 - n_env) * h  =>  h = Δφ * λ / (2π*(n_sio2-n_env))
#     若你要复现实验论文给的具体 nm，请优先用 phase_to_height_sio2_paper() 的离散表。
#     """
#     denom = 2 * torch.pi * (n_sio2 - n_env)
#     return phi * lambda0 / denom
#
#
# def export_quantized_heights_from_model(model: torch.nn.Module, device=None):
#     """
#     从训练好的 model 导出每层相位\(\rightarrow\)四级相位\(\rightarrow\)高度(nm)。
#     返回：list[np.ndarray]，每个元素形状 [N_pixels, N_pixels]（float32，单位 nm）。
#     """
#     heights = []
#     for name, p in model.named_parameters():
#         if not name.startswith("phase_"):
#             continue
#         phase_param = p.detach()
#         if device is not None:
#             phase_param = phase_param.to(device)
#
#         phi = constrain_phase_0_pi(phase_param)
#         phi_q = quantize_phase_4level(phi)
#         h_nm = phase_to_height_sio2_paper(phi_q)  # nm
#         heights.append(h_nm.cpu().numpy().astype(np.float32))
#     return heights
T_OXIDE = 519e-9  # SiO₂ 总膜厚 (m)

def phase_to_height(phi, lambda0=830e-9, n_sio2=1.46):
    return phi * lambda0 / (2 * np.pi * (n_sio2 - 1))

def phase_to_etch_depth(phi_rad, t_oxide=T_OXIDE, phi_max=3*np.pi/4):
    """
    将相位(弧度)转换为刻蚀深度(米)。
    φ=0 → etch=519nm(全刻), φ=3π/4 → etch=0(不刻)
    """
    phi_clamped = np.clip(phi_rad, 0, phi_max)
    return t_oxide * (1.0 - phi_clamped / phi_max)

def quantize_phase_4level(phi_rad):
    """将连续相位量化为 4 级: {0, π/4, π/2, 3π/4}"""
    levels = torch.tensor([0, np.pi/4, np.pi/2, 3*np.pi/4])
    diff = torch.abs(phi_rad[..., None] - levels)
    return levels[torch.argmin(diff, dim=-1)]

# 验证 Fig. S1 映射关系
print("验证 Fig. S1 映射关系:")
for phi, expected in [(0, 519), (np.pi/4, 346), (np.pi/2, 173), (3*np.pi/4, 0)]:
    computed = phase_to_etch_depth(phi) * 1e9
    print(f"  φ={phi/np.pi:.2f}π → etch={computed:.0f} nm (期望 {expected} nm) {'✓' if abs(computed-expected)<1 else '✗'}")
# %% [markdown]
# ### ***Saving***
# %%
torch.save(model, 'MNIST_visible wavelength.pth')
# %%
# 释放显存
torch.cuda.empty_cache()
# %% [markdown]
# ### ***Loading***
# %%
model = torch.load('MNIST_visible wavelength.pth', weights_only=False)
# %% [markdown]
# ### ***Data Analysis***
# %%
for name, param in model.named_parameters():
    if name.find('distance') != -1:
        print('{:.8f}'.format(param.cpu().detach().numpy()[0]))
# %%
for name, param in model.named_parameters():
    if param.requires_grad == True:
        print("\t",name)
# %%
# 查看
plt.rcParams["figure.figsize"] = (15, 4.5)
def visualize(image, label):
    image_E = torch.sqrt(image)
    out = model(image_E.to(device))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout=True)
    ax1.imshow(image.squeeze(), interpolation='none')
    ax1.set_title(f'Input image with\n total intensity {image.squeeze().sum():.2f}')
    output_image = torch.abs(out[1].squeeze()).detach().cpu()
    ax2.imshow(output_image*det_ideal, interpolation='none')
    ax2.set_title(f'Output image with\n total intensity {output_image.squeeze().sum():.2f}')
    dist = out[0].squeeze().detach().cpu()
    ax3.bar(range(len(dist)), dist)
    fig.suptitle("label={}".format(label))
    plt.show()

def mask_visualiztion():
    for name, mask in model.named_parameters():
        if "phase" in name:
            # mask -> sigmoid -> [0,1]，再映射到 [0, π]（rad）
            phi = torch.pi * torch.sigmoid(mask.detach().cpu())

            plt.figure()
            im = plt.imshow(
                phi,
                interpolation="none",
                vmin=0.0,
                vmax=float(torch.pi),
                cmap="twilight",
            )
            plt.title(f"Mask of layer {name} (phase: 0..π rad)")
            cbar = plt.colorbar(im)
            cbar.set_label("Phase (rad)")
            plt.show()
# %%
mask_visualiztion()
# %%
rand_ind = np.random.choice(range(len(val_dataset)), size=10, replace=False)
for ind in rand_ind:
    visualize(val_dataset[ind][0], val_dataset[ind][1])
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------------------- 可改参数 ----------------------
layer_index = 0          # 想看第几层：0,1,2,...
pixel_size_um = 2.0      # 像素尺寸（um），例如 2e-6 m -> 2.0 um

# 导出参数
save_height_map = True
height_map_filename = "height_map.npy"   # 将保存到当前工作目录
save_unit = "m"  # "m" 或 "nm"；建议 "m" 以直接用于 Lumerical
# -----------------------------------------------------


def _get_project_dir() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def _collect_all_layer_etch_depths(model) -> np.ndarray:
    """
    从 model.named_parameters() 中收集 phase_0, phase_1, ... 的刻蚀深度图。

    流程: 裸参数 → π×sigmoid → 连续相位 → 4级量化 → Fig.S1 刻蚀深度

    返回：etch_depth_map, shape = [num_layer, N, N], dtype=float32
          单位：nm（内部计算用nm，保存时再转换）
    """
    params = dict(model.named_parameters())
    phase_keys = [k for k in params.keys() if k.startswith("phase_")]

    def _idx(k: str) -> int:
        return int(k.split("_", 1)[1])

    phase_keys = sorted(phase_keys, key=_idx)
    if not phase_keys:
        raise ValueError("在 model.named_parameters() 中未找到形如 phase_0 的参数。")

    etch_depths = []
    for k in phase_keys:
        # ① 取出裸参数
        raw_param = params[k].detach().cpu()

        # ② 还原 forward 中的相位计算: φ = π × sigmoid(param)
        phi_continuous = np.pi * torch.sigmoid(raw_param)  # (0, π)

        # ③ 量化为 4 级: {0, π/4, π/2, 3π/4}
        phi_quantized = quantize_phase_4level(phi_continuous)

        # ④ 转换为刻蚀深度（米）
        etch_m = phase_to_etch_depth(phi_quantized.numpy())  # [N,N] 单位: m

        # 转为 nm 存储
        etch_nm = (etch_m * 1e9).astype(np.float32)  # [N,N] 单位: nm
        etch_depths.append(etch_nm)

    etch_depth_map_nm = np.stack(etch_depths, axis=0)  # [L,N,N]
    return etch_depth_map_nm


# 1) 生成全层刻蚀深度图（nm）
etch_depth_map_nm = _collect_all_layer_etch_depths(model)

# 2) 保存为 .npy
if save_height_map:
    project_dir = _get_project_dir()
    out_path = os.path.join(project_dir, height_map_filename)

    if save_unit == "m":
        height_map_to_save = (etch_depth_map_nm * 1e-9).astype(np.float32)  # nm -> m
    elif save_unit == "nm":
        height_map_to_save = etch_depth_map_nm.astype(np.float32)
    else:
        raise ValueError("save_unit 只能是 'm' 或 'nm'。")

    np.save(out_path, height_map_to_save)
    print(f"Saved height_map to: {out_path}")
    print(f"  shape={height_map_to_save.shape}, dtype={height_map_to_save.dtype}, unit={save_unit}")
    print(f"  min={float(height_map_to_save.min())}, max={float(height_map_to_save.max())}")

# 3) 同时导出训练好的距离参数
distances = []
num_layers = etch_depth_map_nm.shape[0]
for i in range(num_layers + 2):
    d = getattr(model, f"distance_{i}").detach().cpu().item()
    distances.append(d)
    print(f"distance_{i} = {d*1e3:.4f} mm")
np.save(os.path.join(_get_project_dir(), "trained_distances.npy"), np.array(distances))
print("已保存 trained_distances.npy")

# 4) 取出想看的那一层，用于 2D/3D 可视化（nm）
if not (0 <= layer_index < etch_depth_map_nm.shape[0]):
    raise IndexError(f"layer_index={layer_index} 超出范围 0..{etch_depth_map_nm.shape[0]-1}")

H = etch_depth_map_nm[layer_index]  # [N,N] nm

# 2D 刻蚀深度热力图
plt.figure(figsize=(6, 5))
im = plt.imshow(H, cmap="viridis", origin="upper")
plt.title(f"Quantized etch depth (layer {layer_index}) [nm]")
plt.axis("off")
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label("Etch depth (nm)")
plt.show()

# 刻蚀深度台阶统计
vals, counts = np.unique(np.round(H).astype(int), return_counts=True)
print("Etch depth levels (nm) and pixel counts:")
print("  期望值: 0, 173, 346, 519 nm")
for v, c in zip(vals, counts):
    print(f"  {v} nm: {c} pixels")

# 3D 曲面图
Ny, Nx = H.shape
x = np.arange(Nx, dtype=np.float32) * float(pixel_size_um)
y = np.arange(Ny, dtype=np.float32) * float(pixel_size_um)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, H, rstride=1, cstride=1, cmap="viridis", linewidth=0, antialiased=False)
ax.set_title(f"3D etch depth surface (layer {layer_index})")
ax.set_xlabel("x (um)")
ax.set_ylabel("y (um)")
ax.set_zlabel("Etch depth (nm)")
plt.show()
# %%
# 安全且稳健的混淆矩阵生成与绘制
def update_confusion_matrix(conf_matrix, predicted, labels):
    """
    conf_matrix: torch.Tensor (CPU, integer) shape [C, C]
    predicted: 1D torch Tensor on CPU (predicted labels)
    labels: 1D torch Tensor on CPU (true labels)
    """
    # 转为 Python 列表进行安全索引（也可用张量索引，但此处简单明了）
    pred_list = predicted.tolist()
    label_list = labels.tolist()
    for p, t in zip(pred_list, label_list):
        conf_matrix[p, t] += 1
    return conf_matrix

# 1) 初始化混淆矩阵（在 CPU 上，整数类型）
conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)  # [pred, true]

# 2) 遍历验证集并更新混淆矩阵
with torch.no_grad():
    for i, (images, labels) in enumerate(val_dataloader):
        images = images.to(device)
        images = torch.sqrt(torch.squeeze(images))
        labels = labels.to(device)

        out_labels, out_images = model(images)             # out_labels: logits 或概率 [B, C]
        _, predicted = torch.max(out_labels.data, 1)      # predicted on device

        # 把 predicted 和 labels 转回 CPU 再更新矩阵
        predicted_cpu = predicted.cpu()
        labels_cpu = labels.cpu()

        conf_matrix = update_confusion_matrix(conf_matrix, predicted_cpu, labels_cpu)

# 3) 显示 / 分析混淆矩阵
import numpy as np
cm = conf_matrix.numpy()  # 转为 numpy 方便显示与计算

print("Confusion matrix (rows=predicted, cols=true):")
print(cm)

# 每类的样本数（真实标签统计）
per_class_counts = cm.sum(axis=0)  # 按列求和 -> 每个真实类的样本总数
corrects = np.diag(cm)
per_class_accuracy = (corrects / (per_class_counts + 1e-12)) * 100.0  # 百分比，避免除零

print("每种标签总个数：", per_class_counts.tolist())
print("每种标签预测正确的个数：", corrects.tolist())
print("每种标签的识别准确率为(%):", ["{:.1f}".format(x) for x in per_class_accuracy.tolist()])

# 4) 可视化（matplotlib）
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 5))
im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar(im, fraction=0.046, pad=0.04)

# x/y ticks: true classes (cols) / predicted classes (rows)
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, range(num_classes), rotation=45)
plt.yticks(tick_marks, range(num_classes))

# 在格子里写数值
thresh = cm.max() / 2.0
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, int(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel('Predicted label')
plt.xlabel('True label')
plt.tight_layout()
plt.show()
# %%
# 将模型中的phase导出来
phase = []
for param in model.named_parameters():
    phase.append(param[1])
# %%
phase[0].shape