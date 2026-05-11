import os
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# NEW: 标准HG模式库
from lasy.profiles.transverse.hermite_gaussian_profile import HermiteGaussianTransverseProfile


# ===================== 基本参数 ===================== #
IMG_SIZE = 128
NUM_SAMPLES_PER_CLASS = 6000
OUT_DIR = "data/vcsel_near_synth"
TRAIN_RATIO = 0.8

CLASSES = [
    "0_fundamental",
    "1_first_order",
    "2_second_order",
    "3_third_order",
    "4_fourth_order",
]

USE_RGB_MAGENTA_STYLE = False
GAMMA_CAMERA = (0.75, 1.15)
SATURATION_PROB = 0.60
VIGNETTE_STRENGTH = (0.06, 0.22)

# HG物理参数（可调）
WAVELENGTH = 0.85e-6   # m，VCSEL可按你的器件改
W0X_BASE = 10e-6       # m
W0Y_BASE = 15e-6       # m
XY_RANGE = 30e-6       # m, 对应示例代码[-30,30]um


# ===================== 工具函数 ===================== #
def make_coord_grid(n):
    x = np.linspace(-1, 1, n, dtype=np.float32)
    y = np.linspace(-1, 1, n, dtype=np.float32)
    return np.meshgrid(x, y)

def random_blur(I, sigma_range=(0.0, 1.0)):
    sigma = np.random.uniform(*sigma_range)
    if sigma < 0.08:
        return I
    return gaussian_filter(I.astype(np.float32), sigma=sigma)

def add_vignette(I, xx, yy):
    c = np.random.uniform(*VIGNETTE_STRENGTH)
    vig = np.clip(1.0 - c * (xx**2 + yy**2), 0.6, 1.05)
    return I * vig

def add_low_freq_background(xx, yy):
    ax = np.random.uniform(-0.08, 0.08)
    ay = np.random.uniform(-0.08, 0.08)
    plane = np.random.uniform(0.02, 0.06) + ax * xx + ay * yy
    kx, ky = np.random.uniform(0.5, 2.0, 2)
    wav = 0.015 * (np.sin(kx*np.pi*xx + np.random.uniform(0, 2*np.pi))
                   + np.sin(ky*np.pi*yy + np.random.uniform(0, 2*np.pi)))
    return np.clip(plane + wav, 0, None).astype(np.float32)

def apply_camera_response(I):
    gamma = np.random.uniform(*GAMMA_CAMERA)
    I = np.clip(I, 0, None)
    if I.max() > 0:
        I = I / I.max()
    I = np.power(I, gamma)
    if np.random.rand() < SATURATION_PROB:
        thr = np.random.uniform(0.72, 0.92)
        gain = np.random.uniform(1.05, 1.5)
        I = I * gain
        I = np.where(I > thr, thr + (I - thr) * np.random.uniform(0.05, 0.20), I)
    return np.clip(I, 0, 1).astype(np.float32)

def add_noise(I, gauss_sigma=0.010, speckle_strength=0.18):
    I = I.astype(np.float32)
    I += np.random.normal(0, 0.012 * np.sqrt(np.clip(I, 0, None) + 1e-6), I.shape).astype(np.float32)
    if gauss_sigma > 0:
        I += np.random.normal(0, gauss_sigma, I.shape).astype(np.float32)
    I = np.clip(I, 0, None)
    if speckle_strength > 0:
        speckle = np.random.rayleigh(1.0, I.shape).astype(np.float32)
        I = (1.0 - speckle_strength) * I + speckle_strength * I * speckle
    I = np.clip(I, 0, None)
    if I.max() > 0:
        I /= I.max()
    return I.astype(np.float32)

def rotate_image(I, angle_deg):
    im = Image.fromarray(I.astype(np.float32), mode="F")
    return np.array(im.rotate(angle_deg, resample=Image.BILINEAR, fillcolor=0.0), dtype=np.float32)

def to_magenta_rgb(I):
    I = np.clip(I, 0, 1).astype(np.float32)
    R = np.clip(1.10 * I + 0.10 * (I**0.5), 0, 1)
    G = np.clip(0.12 * (I**1.2), 0, 1)
    B = np.clip(0.85 * I + 0.18 * (I**0.8), 0, 1)
    return (np.stack([R, G, B], axis=-1) * 255).astype(np.uint8)


# ===================== HG 模态生成（替换原LG-like） ===================== #
def class_to_mn(cls_name):
    # 只取“沿x方向”的0~4阶，图样和你给图一致（从单瓣到多瓣）
    mapping = {
        "0_fundamental": (0, 0),
        "1_first_order": (1, 0),
        "2_second_order": (1, 1),
        "3_third_order": (2, 1),
        "4_fourth_order": (2, 2),
    }
    return mapping.get(cls_name, (0, 0))

def gen_hg_mode_intensity(cls_name):
    m, n = class_to_mn(cls_name)

    # 图三更接近圆斑：w0x≈w0y
    w0 = 11e-6 * np.random.uniform(0.92, 1.08)
    w0x = w0 * np.random.uniform(0.98, 1.02)
    w0y = w0 * np.random.uniform(0.98, 1.02)

    wavelength = 0.85e-6 * np.random.uniform(0.99, 1.01)

    xy = np.linspace(-24e-6, 24e-6, IMG_SIZE, dtype=np.float64)
    X, Y = np.meshgrid(xy, xy)

    tp = HermiteGaussianTransverseProfile(
        w_0x=w0x, w_0y=w0y, m=m, n=n, wavelength=wavelength
    )
    E = tp.evaluate(X, Y)
    I = np.abs(E) ** 2
    I = np.asarray(I, dtype=np.float32)

    # 轻微软化，减少“理想数学边界”
    I = gaussian_filter(I, sigma=np.random.uniform(0.8, 1.6))

    # 中心填充一点（实拍常见）
    rr = (X**2 + Y**2) / (2 * (0.30 * w0) ** 2)
    I += np.random.uniform(0.04, 0.10) * np.exp(-rr).astype(np.float32)

    if I.max() > 0:
        I /= I.max()
    return I

def gen_vcsel_nearfield(xx, yy, cls_name):
    I = gen_hg_mode_intensity(cls_name)

    # 图三可有小角度旋转，不要0~360那种大旋转
    I = rotate_image(I, np.random.uniform(-25, 25))

    # 轻微散焦
    I = random_blur(I, sigma_range=(0.6, 1.8))

    # 暗背景+低频偏置
    I = I + add_low_freq_background(xx, yy) * np.random.uniform(0.15, 0.35)
    I = add_vignette(I, xx, yy)

    # 噪声稍强一些，更像手机拍摄
    I = add_noise(I, gauss_sigma=0.010, speckle_strength=0.12)

    # 强一点的相机压缩与饱和
    I = apply_camera_response(I)

    return I.astype(np.float32)

def synthesize_sample(cls_name, xx, yy):
    return gen_vcsel_nearfield(xx, yy, cls_name)


# ===================== 展示示例图 ===================== #
def show_examples(xx, yy, n_examples=8):
    fig, axes = plt.subplots(len(CLASSES), n_examples,
                             figsize=(2*n_examples, 2*len(CLASSES)), squeeze=False)
    for row, cls in enumerate(CLASSES):
        for col in range(n_examples):
            I = synthesize_sample(cls, xx, yy)
            if USE_RGB_MAGENTA_STYLE:
                axes[row][col].imshow(to_magenta_rgb(I))
            else:
                # 更接近你示例风格可改 bone_r
                axes[row][col].imshow(I, cmap="bone_r", vmin=0, vmax=1)
            axes[row][col].axis("off")
            if col == 0:
                axes[row][col].set_title(cls, fontsize=9, loc="left")
    plt.suptitle("VCSEL near Synthetic Dataset - HG Modes Examples", fontsize=13, y=1.01)
    plt.tight_layout()
    preview_path = Path(OUT_DIR) / "preview_near_examples.png"
    fig.savefig(str(preview_path), dpi=150, bbox_inches="tight")
    print(f"Preview saved to: {preview_path}")
    plt.show()


# ===================== 主逻辑 ===================== #
def main():
    np.random.seed(42)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    xx, yy = make_coord_grid(IMG_SIZE)

    # ===== 临时禁用：数据集生成（train/val） =====
    # for split in ["train", "val"]:
    #     for cls in CLASSES:
    #         (Path(OUT_DIR) / split / cls).mkdir(parents=True, exist_ok=True)
    #
    # for cls in CLASSES:
    #     n_train = int(NUM_SAMPLES_PER_CLASS * TRAIN_RATIO)
    #     n_val = NUM_SAMPLES_PER_CLASS - n_train
    #     print(f"[near] Generating '{cls}': {n_train} train + {n_val} val")
    #
    #     for i in range(n_train):
    #         I = synthesize_sample(cls, xx, yy)
    #         p = Path(OUT_DIR) / "train" / cls / f"{cls}_{i:05d}.png"
    #         if USE_RGB_MAGENTA_STYLE:
    #             Image.fromarray(to_magenta_rgb(I), "RGB").save(p)
    #         else:
    #             Image.fromarray((I * 255).astype(np.uint8), "L").save(p)
    #
    #     for i in range(n_val):
    #         I = synthesize_sample(cls, xx, yy)
    #         p = Path(OUT_DIR) / "val" / cls / f"{cls}_{i:05d}.png"
    #         if USE_RGB_MAGENTA_STYLE:
    #             Image.fromarray(to_magenta_rgb(I), "RGB").save(p)
    #         else:
    #             Image.fromarray((I * 255).astype(np.uint8), "L").save(p)

    # ===== 只生成示例图 =====
    print("Dataset generation is disabled. Only generating preview...")
    show_examples(xx, yy, n_examples=8)

if __name__ == "__main__":
    main()