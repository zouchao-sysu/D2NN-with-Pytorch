import os
import math
import numpy as np
from pathlib import Path
from PIL import Image

# ===================== 基本参数 ===================== #
IMG_SIZE = 128
NUM_SAMPLES_PER_CLASS = 6000
OUT_DIR = "data/vcsel_synth"
TRAIN_RATIO = 0.8

CLASSES = [
    "0 fundamental",
    "1 two_lobes",
    "2 four_lobes",
    "3 multi_lobes"
]

# ===================== 工具函数 ===================== #

def make_coord_grid(n):
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(x, y)
    return xx, yy

def rotate_image(I, angle_deg):
    im = Image.fromarray(I.astype(np.float32))
    im_rot = im.rotate(angle_deg, resample=Image.BILINEAR)
    return np.array(im_rot, dtype=np.float32)

def add_noise_and_normalize(I, gauss_sigma=0.01, speckle_strength=0.2):
    I = I.astype(np.float32)
    if gauss_sigma > 0:
        I += np.random.normal(0.0, gauss_sigma, size=I.shape).astype(np.float32)
    I = np.clip(I, 0, None)

    if speckle_strength > 0:
        speckle = np.random.rayleigh(scale=1.0, size=I.shape).astype(np.float32)
        I = (1.0 - speckle_strength) * I + speckle_strength * I * speckle

    I = I - I.min()
    if I.max() > 0:
        I = I / I.max()
    return I

# ===================== 小光斑模式 ===================== #

def flower_spot(xx, yy, x0, y0, w, mode_type="0 fundamental"):
    X = xx - x0
    Y = yy - y0
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    env = np.exp(-(r**2) / (2 * w**2))

    if mode_type == "0 fundamental":
        I = env
    elif mode_type == "1 two_lobes":
        I = env * (0.6 + 0.4 * np.cos(2 * theta))**2
    elif mode_type == "2 four_lobes":
        I = env * (0.6 + 0.4 * np.cos(4 * theta))**2
    elif mode_type == "3 multi_lobes":
        k = np.random.randint(6, 10)
        phi = np.random.uniform(0, 2*np.pi)
        I = env * (0.4 + 0.6 * np.cos(k * theta + phi))**2
    else:
        I = env

    return I

# ===================== 中心强斑 ===================== #

def gen_center_spot(xx, yy):
    w0 = np.random.uniform(0.12, 0.20)
    x0 = np.random.uniform(-0.05, 0.05)
    y0 = np.random.uniform(-0.05, 0.05)

    I_center = np.exp(-((xx-x0)**2 + (yy-y0)**2) / (2 * w0**2))

    if np.random.rand() < 0.3:
        dx = np.random.uniform(0.05, 0.10)
        dy = np.random.uniform(-0.03, 0.03)
        I_center2 = np.exp(-((xx-(x0+dx))**2 + (yy-(y0+dy))**2) / (2 * (w0*0.8)**2))
        I_center = I_center + I_center2

    scale = np.random.uniform(5.0, 8.0)
    return scale * I_center

# ===================== 晶格小光斑 ===================== #

def gen_lattice_spots(xx, yy, mode_type):
    I = np.zeros_like(xx, dtype=np.float32)

    pitch = np.random.uniform(0.25, 0.35)
    nx = np.random.randint(3, 5)
    ny = np.random.randint(3, 5)

    x_off = np.random.uniform(-0.1, 0.1)
    y_off = np.random.uniform(-0.1, 0.1)

    for i in range(-nx, nx+1):
        for j in range(-ny, ny+1):
            if np.random.rand() < 0.15:
                continue

            x0 = i * pitch + x_off
            y0 = j * pitch + y_off

            if x0**2 + y0**2 < 0.10:
                continue

            w = np.random.uniform(0.04, 0.06)
            I += flower_spot(xx, yy, x0, y0, w, mode_type=mode_type)

    return I

# ===================== 合成模式 ===================== #

def gen_vcsel_style_mode(xx, yy, cls_name):
    I_center = gen_center_spot(xx, yy)

    # 基模：只保留中心斑 + 背景
    if cls_name == "0 fundamental":
        I = I_center + np.random.rand(*xx.shape).astype(np.float32) * np.random.uniform(0.01, 0.03)
        return add_noise_and_normalize(I, gauss_sigma=0.01, speckle_strength=0.1)

    # 其它模式：中心 + 晶格小光斑
    if cls_name == "1 two_lobes":
        lattice_mode = "1 two_lobes"
    elif cls_name == "2 four_lobes":
        lattice_mode = "2 four_lobes"
    elif cls_name == "3 multi_lobes":
        lattice_mode = "3 multi_lobes"
    else:
        lattice_mode = "0 fundamental"

    I_lattice = gen_lattice_spots(xx, yy, lattice_mode)
    I_bg = np.random.rand(*xx.shape).astype(np.float32) * np.random.uniform(0.02, 0.06)

    I = I_center + 0.8 * I_lattice + I_bg
    angle = np.random.uniform(-10, 10)
    I = rotate_image(I, angle)

    return add_noise_and_normalize(I, gauss_sigma=0.01, speckle_strength=0.2)

def synthesize_sample(cls_name, xx, yy):
    return gen_vcsel_style_mode(xx, yy, cls_name)

# ===================== 主逻辑 ===================== #

def main():
    np.random.seed(42)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    xx, yy = make_coord_grid(IMG_SIZE)

    for split in ["train", "val"]:
        for cls in CLASSES:
            out_dir = Path(OUT_DIR) / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        n_total = NUM_SAMPLES_PER_CLASS
        n_train = int(n_total * TRAIN_RATIO)
        n_val = n_total - n_train

        print(f"Generating class '{cls}': {n_train} train + {n_val} val")

        for i in range(n_train):
            I = synthesize_sample(cls, xx, yy)
            img_uint8 = (I * 255).astype(np.uint8)
            im = Image.fromarray(img_uint8, mode="L")
            out_path = Path(OUT_DIR) / "train" / cls / f"{cls}_{i:05d}.png"
            im.save(out_path)

        for i in range(n_val):
            I = synthesize_sample(cls, xx, yy)
            img_uint8 = (I * 255).astype(np.uint8)
            im = Image.fromarray(img_uint8, mode="L")
            out_path = Path(OUT_DIR) / "val" / cls / f"{cls}_{i:05d}.png"
            im.save(out_path)

    print("Done. Dataset saved to:", OUT_DIR)

if __name__ == "__main__":
    main()